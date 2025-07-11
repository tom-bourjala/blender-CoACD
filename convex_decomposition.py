import pathlib
import random
import subprocess
import tempfile
import threading
import queue
import re  # Import regex module
from pathlib import Path
from typing import List, Tuple

import bpy  # type: ignore
import bpy_types  # type: ignore
import bmesh
from mathutils import Vector, Matrix

bl_info = {
    'name': 'CoACD Convex Decomposition',
    'blender': (4, 2, 0),
    'category': 'Object',
    'version': (1, 0, 1),
    'author': 'Tom Bourjala',
    'description': 'Create collision shapes using CoACD',
    'warning': 'WIP',
}


class ConvexDecompositionPreferences(bpy.types.AddonPreferences):
    """Addon preferences menu."""
    bl_idname = "convex_decomposition"

    coacd_binary: bpy.props.StringProperty(  # type: ignore
        name="CoACD Binary",
        subtype='FILE_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "coacd_binary")


class SelectionGuard():
    """Ensure the same objects are selected at the end."""
    def __init__(self, clear: bool = False):
        self.clear = clear
        self.selected = None
        self.active = None

    def __enter__(self, clear=False):
        self.selected = bpy.context.selected_objects
        self.active = bpy.context.view_layer.objects.active

        if self.clear:
            bpy.ops.object.select_all(action='DESELECT')
        return self

    def __exit__(self, *args, **kwargs):
        bpy.ops.object.select_all(action='DESELECT')
        assert self.selected is not None
        assert self.active is not None
        for obj in self.selected:
            obj.select_set(True)

        # Restore the active object.
        bpy.context.view_layer.objects.active = self.active


class ConvexDecompositionBaseOperator(bpy.types.Operator):
    """Base class for the operators with common utility methods."""

    bl_idname = 'opr.convex_decomposition_base'
    bl_label = 'Convex Decomposition Base Class'

    def get_selected_object(self) -> Tuple[bpy_types.Object, bool]:
        """Return the selected object.

        Set the error flag if more or less than one object is currently
        selected or if we are not in OBJECT mode.

        """
        # User must be in OBJECT mode.
        if bpy.context.mode != 'OBJECT':
            self.report({'ERROR'}, "Must be in OBJECT mode")
            return None, True

        # User must have exactly one object selected.
        selected = bpy.context.selected_objects
        if len(selected) != 1:
            self.report({'ERROR'}, "Must have exactly one object selected")
            return None, True

        return selected[0], False

    def remove_stale_hulls(self, root_obj: bpy_types.Object) -> None:
        """Remove the convex decomposition results from previous runs for `root_obj`."""
        # Store current mode
        current_mode = bpy.context.object.mode

        # Switch to object mode if not already
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        with SelectionGuard(clear=True):
            for obj in bpy.data.objects:
                if obj.name.startswith(f"UCX_{root_obj.name}_"):
                    obj.select_set(True)
            bpy.ops.object.delete()

        # Restore original mode
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=current_mode)

    def rename_hulls(self, parent: bpy.types.Object, collection_name: str) -> None:
        """Rename all convex hulls in the specified collection to Unreal Engine format.

        Renames hulls to the format "UCX_{parent.name}_{seq-number}".
        This ensures that Unreal Engine can load the object and automatically
        recognize all its collision shapes.
        """
        hull_collection = bpy.data.collections.get(collection_name)
        if not hull_collection:
            self.report({'WARNING'}, f"Collection '{collection_name}' not found. Skipping hull renaming.")
            return

        # Step 1: Rename to temporary names
        temp_hulls = []
        for obj in hull_collection.objects:
            if obj.parent == parent:
                obj.name = "TEMP_HULL"
                temp_hulls.append(obj)

        # Step 2: Rename to final names
        for i, hull_obj in enumerate(temp_hulls):
            hull_obj.name = f"UCX_{parent.name}_{i}"

    def upsert_collection(self, name: str) -> bpy.types.Collection:
        """Create a dedicated collection` `name` for the convex hulls.

        Does nothing if the collection already exists.

        """
        try:
            collection = bpy.data.collections[name]
        except KeyError:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def randomise_colour(self, obj: bpy.types.Object) -> None:
        """Assign a random colour to `obj` with fixed transparency of 90%."""
        red, green, blue = [random.random() for _ in range(3)]

        material = bpy.data.materials.new("random material")
        transparency = 90
        material.diffuse_color = (red, green, blue, (100 - transparency) / 100.0)
        obj.data.materials.clear()
        obj.data.materials.append(material)


class ConvexDecompositionRunOperator(ConvexDecompositionBaseOperator):
    """Use CoACD to create a convex decomposition of objects."""
    bl_idname = 'opr.convex_decomposition_run'
    bl_label = 'Run CoACD Convex Decomposition'
    bl_description = "Run CoACD Solver"

    # Store process and other data for modal operation
    process = None
    output_lines = []
    error_lines = []
    obj_file = None
    tmp_path = None
    hull_path = None
    parent_obj = None
    hull_objs = []
    is_split_part = False
    original_name = ""

    stdout_queue = None
    stderr_queue = None
    stdout_thread = None
    stderr_thread = None

    instance = None  # Class variable to hold the instance of the operator

    def export_mesh_for_solver(self, obj: bpy.types.Object, path: Path) -> Path:
        """Save a temporary copy of `obj` in OBJ format to a temporary folder.

        This is necessary because the various solvers all expect an OBJ file as input.
        """
        fname = path / "src.obj"

        # Store the current selection state
        original_selection = bpy.context.selected_objects
        original_active = bpy.context.view_layer.objects.active

        try:
            # Ensure only the desired object is selected
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Export the object
            bpy.ops.wm.obj_export(
                filepath=str(fname),
                check_existing=False,
                export_selected_objects=True,
                export_triangulated_mesh=True,
                export_materials=False,
                apply_modifiers=True
            )
        finally:
            # Restore the original selection state
            bpy.ops.object.select_all(action='DESELECT')
            for o in original_selection:
                o.select_set(True)
            bpy.context.view_layer.objects.active = original_active

        return fname

    def enqueue_output(self, out, queue):
        """Function to read output in a separate thread and put it in a queue."""
        for line in iter(out.readline, ''):
            queue.put(line)
        out.close()

    def start_process(self, cmd, cwd):
        """Start the external process asynchronously."""
        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Create queues and threads for stdout and stderr
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()

        self.stdout_thread = threading.Thread(target=self.enqueue_output, args=(self.process.stdout, self.stdout_queue))
        self.stderr_thread = threading.Thread(target=self.enqueue_output, args=(self.process.stderr, self.stderr_queue))

        self.stdout_thread.daemon = True  # Thread dies with the program
        self.stderr_thread.daemon = True

        self.stdout_thread.start()
        self.stderr_thread.start()

    def modal(self, context, event):
        if event.type == 'TIMER':
            # Read stdout
            while not self.stdout_queue.empty():
                line = self.stdout_queue.get_nowait()
                if line:
                    self.report({'INFO'}, line.strip())

                    # Parse iteration info
                    iter_match = re.search(r'iter (\d+) ---- waiting pool: (\d+)', line)
                    if iter_match:
                        iter_num = int(iter_match.group(1))
                        waiting_pool = int(iter_match.group(2))
                        # Update properties (add +1 to compensate for id starting at 0)
                        context.scene.ConvDecompProperties.current_iteration = iter_num + 1
                        context.scene.ConvDecompProperties.waiting_pool = waiting_pool

                        # Force UI redraw
                        for window in context.window_manager.windows:
                            for area in window.screen.areas:
                                if area.type == 'VIEW_3D':
                                    area.tag_redraw()

                    # Parse processing percentage
                    percent_match = re.search(r'Processing \[(\d+\.?\d*)%\]', line)
                    if percent_match:
                        percent = float(percent_match.group(1))
                        context.scene.ConvDecompProperties.processing_percentage = percent

                        # Force UI redraw
                        for window in context.window_manager.windows:
                            for area in window.screen.areas:
                                if area.type == 'VIEW_3D':
                                    area.tag_redraw()

            # Read stderr
            while not self.stderr_queue.empty():
                line = self.stderr_queue.get_nowait()
                if line:
                    self.report({'ERROR'}, line.strip())

            # Check if process has finished
            if self.process.poll() is not None:
                # Clean up threads
                self.stdout_thread.join()
                self.stderr_thread.join()

                # Import solver results
                try:
                    self.import_solver_results(self.hull_path, context.scene.ConvDecompProperties.tmp_hull_prefix)
                except FileNotFoundError:
                    self.report({'ERROR'}, "Solver did not produce output file. Decomposition failed or was cancelled.")
                    # Remove the timer
                    wm = context.window_manager
                    wm.event_timer_remove(self._timer)

                    context.scene.ConvDecompProperties.is_running = False
                    ConvexDecompositionRunOperator.instance = None
                    return {'CANCELLED'}

                self.hull_objs = [obj for obj in bpy.data.objects if obj.name.startswith(context.scene.ConvDecompProperties.tmp_hull_prefix)]

                # Process the resulting hull objects
                self.process_hull_objects(self.hull_objs, self.parent_obj, context.scene.ConvDecompProperties)

                # If the original object was a split part, delete it
                if self.is_split_part:
                    bpy.data.objects.remove(self.parent_obj, do_unlink=True)

                # Rename Hulls
                self.rename_hulls(self.parent_obj, context.scene.ConvDecompProperties.hull_collection_name)

                self.report({'INFO'}, f"Convex decomposition completed successfully for {self.original_name}")

                # Remove the timer
                wm = context.window_manager
                wm.event_timer_remove(self._timer)

                context.scene.ConvDecompProperties.is_running = False
                ConvexDecompositionRunOperator.instance = None

                return {'FINISHED'}
        return {'PASS_THROUGH'}

    def execute(self, context):
        # Reset instance variables
        self.process = None
        self.output_lines = []
        self.error_lines = []
        self.obj_file = None
        self.tmp_path = None
        self.hull_path = None
        self.parent_obj = None
        self.hull_objs = []
        self.is_split_part = False
        self.original_name = ""
        self.stdout_queue = None
        self.stderr_queue = None
        self.stdout_thread = None
        self.stderr_thread = None

        # Reset progress properties
        context.scene.ConvDecompProperties.current_iteration = 0
        context.scene.ConvDecompProperties.waiting_pool = 0
        context.scene.ConvDecompProperties.processing_percentage = 0.0

        # Convenience.
        prefs = context.preferences.addons["convex_decomposition"].preferences
        props = context.scene.ConvDecompProperties

        if context.object.mode != 'OBJECT':
            self.report({'ERROR'}, "Must be in OBJECT mode for CoACD")
            return {'CANCELLED'}

        # Check if exactly one object is selected for CoACD
        selected_objects = context.selected_objects
        if len(selected_objects) != 1:
            self.report({'ERROR'}, "Please select exactly one object for CoACD")
            return {'CANCELLED'}

        root_obj = selected_objects[0]

        if root_obj is None:
            self.report({'ERROR'}, "No active object selected")
            return {'CANCELLED'}

        # Check if the object is already a split part
        self.is_split_part = root_obj.name.startswith("UCX_")
        self.original_name = root_obj.name

        if self.is_split_part:
            self.parent_obj = root_obj.parent
            if not self.parent_obj:
                self.parent_obj = root_obj  # Fallback if no parent
        else:
            self.parent_obj = root_obj

        self.report({'INFO'}, f"Computing collision meshes for <{self.original_name}>")

        if not self.is_split_part:
            self.remove_stale_hulls(root_obj)

        # Save the selected root object to a temporary location for the solver.
        self.tmp_path = Path(tempfile.mkdtemp(prefix="devcomp-"))
        self.report({"INFO"}, f"Created temporary directory for solver: {self.tmp_path}")
        self.obj_file = self.export_mesh_for_solver(root_obj, self.tmp_path)

        # Use CoACD to compute the convex decomposition.
        cmd, self.hull_path = self.get_coacd_command(self.obj_file, context.scene.ConvDecompPropertiesCoACD, Path(prefs.coacd_binary))

        # Start the process
        self.start_process(cmd, self.obj_file.parent)

        # Add a timer to call modal periodically
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        # Set running flag and store instance
        context.scene.ConvDecompProperties.is_running = True
        ConvexDecompositionRunOperator.instance = self

        return {'RUNNING_MODAL'}

    def get_coacd_command(self, obj_file: Path, props: bpy.types.PropertyGroup, binary: Path):
        result_file = obj_file.parent / "hulls.obj"

        cmd = [
            str(binary),
            "--input", str(obj_file),
            "--output", str(result_file),

            "--threshold", str(props.f_threshold),
            "-k", str(props.f_k),

            "--mcts-iteration", str(props.i_mcts_iterations),
            "--mcts-depth", str(props.i_mcts_depth),
            "--mcts-node", str(props.i_mcts_node),
            "--prep-resolution", str(props.i_prep_resolution),
            "--resolution", str(props.i_resolution),
            "--preprocess-mode", props.e_preprocess_mode,
        ]

        if props.b_pca:
            cmd.append("--pca")

        if not props.b_merge:
            cmd.append("--no-merge")
        else:
            cmd.extend(["--max-convex-hull", str(props.i_max_convex_hull)])

        if props.b_decimate:
            cmd.append("--decimate")
            cmd.extend(["--max-ch-vertex", str(props.i_max_ch_vertex)])

        if props.b_extrude:
            cmd.append("--extrude")
            cmd.extend(["--extrude-margin", str(props.f_extrude_margin)])

        if props.i_seed != 0:
            cmd.extend(["--seed", str(props.i_seed)])

        self.report({"INFO"}, f"Running command: {' '.join(cmd)}")
        return cmd, result_file

    def import_solver_results(self, fname: Path, hull_prefix: str):
        """Load the solver output `fname` (an OBJ file)."""
        # Replace all object names in the OBJ file with a solver independent
        # naming scheme.
        data = ""
        lines = fname.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("o "):
                data += f"o {hull_prefix}{i}\n"
            else:
                data += line + "\n"
        fname.write_text(data)

        # Import the hulls back into Blender.
        with SelectionGuard():
            bpy.ops.wm.obj_import(
                filepath=str(fname),
                filter_glob='*.obj',
            )

    def process_hull_objects(self, hull_objs, parent_obj, props):
        hull_collection = self.upsert_collection(props.hull_collection_name)
        for obj in hull_objs:
            # Unlink the current object from all its collections.
            for coll in obj.users_collection:
                coll.objects.unlink(obj)

            # Link the object to our dedicated collection.
            hull_collection.objects.link(obj)

            # Assign a random colour to the hull.
            self.randomise_colour(obj)

            # Parent the hull to the parent object without changing the relative transform.
            if parent_obj:
                obj.parent = parent_obj
                obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

    def cancel(self, context):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self.stdout_thread.join()
            self.stderr_thread.join()
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        context.scene.ConvDecompProperties.is_running = False
        ConvexDecompositionRunOperator.instance = None
        self.report({'INFO'}, "Convex decomposition cancelled.")
        # Clean up temporary files
        if self.tmp_path and self.tmp_path.exists():
            import shutil
            shutil.rmtree(self.tmp_path)


class ConvexDecompositionCancelOperator(bpy.types.Operator):
    bl_idname = 'opr.convex_decomposition_cancel'
    bl_label = 'Cancel Convex Decomposition'
    bl_description = "Cancel CoACD Decomposition"

    def execute(self, context):
        if ConvexDecompositionRunOperator.instance:
            ConvexDecompositionRunOperator.instance.cancel(context)
            ConvexDecompositionRunOperator.instance = None
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No operation running.")
            return {'CANCELLED'}


class ConvexDecompositionPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_ConvDec'
    bl_label = 'Convex Decomposition'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ConvDecomp"

    def draw(self, context):
        props = context.scene.ConvDecompProperties
        prefs = context.preferences.addons["convex_decomposition"].preferences
        layout = self.layout

        solver_props = context.scene.ConvDecompPropertiesCoACD

        is_valid_solver = True
        binary = Path(prefs.coacd_binary)
        is_valid_solver = binary.name != "" and binary.exists()

        # Solver Specific parameters.
        layout.separator()
        box = layout.box()
        box.enabled = is_valid_solver and not props.is_running

        # CoACD specific properties
        box.prop(solver_props, 'f_threshold')
        box.prop(solver_props, 'i_mcts_iterations')
        box.prop(solver_props, 'i_mcts_depth')
        box.prop(solver_props, 'i_mcts_node')
        box.prop(solver_props, 'i_prep_resolution')
        box.prop(solver_props, 'i_resolution')
        box.prop(solver_props, 'f_k')
        box.prop(solver_props, 'e_preprocess_mode')
        box.prop(solver_props, 'b_pca')
        box.prop(solver_props, 'b_merge')

        row = box.row()
        row.enabled = solver_props.b_merge
        row.prop(solver_props, 'i_max_convex_hull')

        box.prop(solver_props, 'b_decimate')

        row = box.row()
        row.enabled = solver_props.b_decimate
        row.prop(solver_props, 'i_max_ch_vertex')

        box.prop(solver_props, 'b_extrude')

        row = box.row()
        row.enabled = solver_props.b_extrude
        row.prop(solver_props, 'f_extrude_margin')

        box.prop(solver_props, 'i_seed')

        # Run or Cancel button
        layout.separator()
        if props.is_running:
            # Display progress bars
            iter_bar = self.get_iteration_progress_bar(props)
            layout.label(text=iter_bar)
            waiting_pool_text = f"Waiting Pool: {props.waiting_pool}"
            layout.label(text=waiting_pool_text)
            percent_bar = self.get_percentage_progress_bar(props)
            layout.label(text=percent_bar)
            row = layout.row()
            row.operator('opr.convex_decomposition_cancel', text="Cancel")
        else:
            row = layout.row()
            row.operator('opr.convex_decomposition_run', text="Run")
            row.enabled = is_valid_solver

    def get_iteration_progress_bar(self, props):
        current = props.current_iteration
        bar = '■' * (current - 1) + '▨'
        return f"Iterations : {bar} {current}"

    def get_percentage_progress_bar(self, props):
        percent = props.processing_percentage
        return f"Progress : {percent:.1f}%"


class ConvexDecompositionProperties(bpy.types.PropertyGroup):
    tmp_hull_prefix: bpy.props.StringProperty(  # type: ignore
        name="Hull Prefix",
        description="Name prefix for the temporary hull names created by the solvers.",
        default="_tmphull_",
    )
    hull_collection_name: bpy.props.StringProperty(  # type: ignore
        name="Hull Collection",
        description="The collection to hold all the convex hulls.",
        default="convex hulls",
    )
    is_running: bpy.props.BoolProperty(
        name="Is Running",
        description="Indicates if the decomposition is currently running.",
        default=False,
    )
    current_iteration: bpy.props.IntProperty(
        name="Current Iteration",
        default=0,
    )
    waiting_pool: bpy.props.IntProperty(
        name="Total Iterations",
        default=0,
    )
    processing_percentage: bpy.props.FloatProperty(
        name="Processing Percentage",
        default=0.0,
    )


class ConvexDecompositionPropertiesCoACD(bpy.types.PropertyGroup):
    f_threshold: bpy.props.FloatProperty(
        name="Concavity Threshold",
        description=(
            "Primary parameter controlling the quality of the decomposition. "
            "A lower value produces more detailed convex hulls but increases computation time. "
            "Range: 0.01 (high detail) to 1.0 (low detail). Default: 0.05"
        ),
        default=0.05,
        min=0.01,
        max=1.0,
        subtype='UNSIGNED'
    )
    i_mcts_iterations: bpy.props.IntProperty(
        name="MCTS Iterations",
        description=(
            "Number of search iterations in Monte Carlo Tree Search (MCTS). "
            "Increasing this may improve decomposition quality but increases computation time. "
            "Range: 60 to 2000. Default: 100"
        ),
        default=100,
        min=60,
        max=2000,
        subtype='UNSIGNED'
    )
    i_mcts_depth: bpy.props.IntProperty(
        name="MCTS Depth",
        description=(
            "Maximum search depth in MCTS. Higher values may produce better cutting strategies "
            "but increase computation time. Range: 2 to 7. Default: 3"
        ),
        default=3,
        min=2,
        max=7,
        subtype='UNSIGNED'
    )
    i_mcts_node: bpy.props.IntProperty(
        name="MCTS Node",
        description=(
            "Maximum number of child nodes in MCTS. Higher values can potentially find better "
            "decompositions but increase computation time. Range: 10 to 40. Default: 20"
        ),
        default=20,
        min=10,
        max=40,
        subtype='UNSIGNED'
    )
    i_prep_resolution: bpy.props.IntProperty(
        name="Manifold Pre-Processing Resolution",
        description=(
            "Resolution used during manifold pre-processing. Higher values produce a mesh closer "
            "to the original but may increase triangle count and computation time. "
            "Range: 20 to 500. Default: 50"
        ),
        default=50,
        min=20,
        max=500,
        subtype='UNSIGNED'
    )
    i_resolution: bpy.props.IntProperty(
        name="Sampling Resolution",
        description=(
            "Sampling resolution used for computing the Hausdorff distance during decomposition. "
            "Higher values can improve accuracy but increase computation time. "
            "Range: 1000 to 10000. Default: 2000"
        ),
        default=2000,
        min=1000,
        max=10000,
        subtype='UNSIGNED'
    )

    f_k: bpy.props.FloatProperty(
        name="K",
        description=(
            "Value of K used in R_v calculation during decomposition. Adjusting this can affect "
            "the balance between concavity and volume in the decomposition process. "
            "Range: 0.0 to 1.0. Default: 0.3"
        ),
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='UNSIGNED'
    )

    e_preprocess_mode: bpy.props.EnumProperty(
        name="Preprocess Mode",
        description=(
            "Choose the manifold pre-processing mode. 'Auto' checks if the input mesh is manifold "
            "and applies pre-processing if necessary. 'On' forces manifold pre-processing. "
            "'Off' skips pre-processing (use only if you are sure your mesh is manifold). "
            "Default: 'Auto'"
        ),
        items=[
            ('auto', 'Auto', 'Automatically check input mesh manifoldness and preprocess if necessary.'),
            ('on', 'On', 'Force turn on the pre-processing regardless of mesh manifoldness.'),
            ('off', 'Off', 'Force turn off the pre-processing. Use only if mesh is manifold'),
        ],
        default='auto',
    )

    b_merge: bpy.props.BoolProperty(
        name="Enable Merge",
        description=(
            "Enable merge post-processing step to reduce the number of convex hulls by merging adjacent parts. "
            "Disabling this may result in more convex hulls. Default is enabled"
        ),
        default=True,
    )
    i_max_convex_hull: bpy.props.IntProperty(
        name="Max Convex Hulls",
        description=(
            "Maximum number of convex hulls allowed in the result. Set to -1 for no limit. "
            "Only effective when 'Enable Merge' is checked. May introduce convex hulls with concavity "
            "larger than the threshold if the limit is reached. Default: -1 (no limit)"
        ),
        default=-1,
        min=-1,
        subtype='UNSIGNED'
    )

    b_decimate: bpy.props.BoolProperty(
        name="Enable Decimate",
        description=(
            "Enable decimation to enforce a maximum number of vertices per convex hull. "
            "Use to reduce complexity of individual convex hulls. Default is disabled"
        ),
        default=False,
    )
    i_max_ch_vertex: bpy.props.IntProperty(
        name="Max Hull Vertices",
        description=(
            "Maximum number of vertices allowed per convex hull. Only effective when 'Enable Decimate' "
            "is checked. Range: 4 and above. Default: 256"
        ),
        default=256,
        min=4,
        subtype='UNSIGNED'
    )

    b_extrude: bpy.props.BoolProperty(
        name="Enable Extrude",
        description=(
            "Enable extrusion of neighboring convex hulls along overlapping faces to improve collision detection. "
            "Default is disabled"
        ),
        default=False,
    )
    f_extrude_margin: bpy.props.FloatProperty(
        name="Extrude Margin",
        description=(
            "Margin distance for extrusion when 'Enable Extrude' is checked. Adjust to control the amount "
            "of extrusion along overlapping faces. Default: 0.01"
        ),
        default=0.01,
        min=0.0,
        subtype='UNSIGNED'
    )

    b_pca: bpy.props.BoolProperty(
        name="Enable PCA Pre-Processing",
        description=(
            "Enable PCA (Principal Component Analysis) pre-processing to align the object for potentially "
            "better decomposition results. Default is disabled"
        ),
        default=False,
    )

    i_seed: bpy.props.IntProperty(
        name="Seed",
        description=(
            "Random seed used for sampling during decomposition. Use 0 for a random seed (non-deterministic results). "
            "Set a specific value for reproducible results. Default: 0"
        ),
        default=0,
        min=0,
        subtype='UNSIGNED'
    )


# ----------------------------------------------------------------------
# Addon registration.
# ----------------------------------------------------------------------


CLASSES = [
    ConvexDecompositionPanel,
    ConvexDecompositionProperties,
    ConvexDecompositionPropertiesCoACD,
    ConvexDecompositionRunOperator,
    ConvexDecompositionCancelOperator,
    ConvexDecompositionPreferences,
]

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ConvDecompProperties = bpy.props.PointerProperty(type=ConvexDecompositionProperties)
    bpy.types.Scene.ConvDecompPropertiesCoACD = bpy.props.PointerProperty(type=ConvexDecompositionPropertiesCoACD)

def unregister():
    for cls in CLASSES:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.ConvDecompProperties
    del bpy.types.Scene.ConvDecompPropertiesCoACD

if __name__ == '__main__':
    register()
