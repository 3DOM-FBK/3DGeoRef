import bpy
import bmesh
import os
import sys
import mathutils
import math
import argparse
import platform
import json
from mathutils import Matrix, Vector


HDRI_PATH = "/app/hdri/studio.exr"


# ===== Function: parse_arguments =====
def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_file (str): Path to the input file (required).
            - output_folder (str): Path to the output folder (required).
            - streetviews (int): Number of streetview-style renderings around the model (default: 5).
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Create synthetic views.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input file')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument("--streetviews", type=int, default=5, help="Number of streetview-style renderings around the model (default: 5)")

    return parser.parse_args(argv)


# ===== Function: clear_scene =====
def clear_scene():
    """
    Clears the Blender scene by deleting all objects, meshes, and materials.  
    This ensures the scene is completely reset and free of leftover data.

    Returns:
        None
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


# ===== Function: import_model =====
def import_model(filepath):
    """
    Imports a 3D model into the Blender scene based on its file extension.  
    Supports `.obj`, `.ply`, `.fbx`, `.glb`, and `.gltf` formats.  
    After import, the first selected object is returned.

    Args:
        filepath (str): Path to the 3D model file.

    Returns:
        bpy.types.Object: The imported Blender object.

    Raises:
        ValueError: If the file format is not supported.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    return bpy.context.selected_objects[0]


# ===== Function: get_combined_bounding_box =====
def get_combined_bounding_box(objects):
    """
    Calculates the combined world-space bounding box of a list of Blender objects.  
    Only objects of type `MESH` are considered.  
    Returns the bounding box center, size, maximum, and minimum coordinates.

    Args:
        objects (list[bpy.types.Object]): List of Blender objects to evaluate.

    Returns:
        tuple:
            - mathutils.Vector: Center of the combined bounding box.  
            - mathutils.Vector: Size (extent along X, Y, Z).  
            - mathutils.Vector: Maximum coordinates of the bounding box.  
            - mathutils.Vector: Minimum coordinates of the bounding box.  
    """
    bbox_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in objects:
        if obj.type != 'MESH':
            continue

        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            bbox_min = mathutils.Vector((
                min(bbox_min.x, world_corner.x),
                min(bbox_min.y, world_corner.y),
                min(bbox_min.z, world_corner.z)
            ))
            bbox_max = mathutils.Vector((
                max(bbox_max.x, world_corner.x),
                max(bbox_max.y, world_corner.y),
                max(bbox_max.z, world_corner.z)
            ))

    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    return center, size, bbox_max, bbox_min


def auto_adjust_camera_clipping(camera_obj, margin=0.1):
    """
    Automatically calculates and sets clip_start and clip_end
    based on the distance of visible objects from the camera.

    Args:
        camera_obj (bpy.types.Object): The camera to adjust.
        margin (float): Additional percentage margin (e.g., 0.1 = 10%).
    """
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        print("⚠️ Nessuna mesh trovata nella scena.")
        return

    cam_loc = camera_obj.matrix_world.translation
    cam_forward = camera_obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))

    min_dist = float('inf')
    max_dist = 0.0

    for obj in meshes:
        world_verts = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

        for v in world_verts:
            vec = v - cam_loc
            dist_along_view = vec.dot(cam_forward)

            if dist_along_view > 0:
                min_dist = min(min_dist, dist_along_view)
                max_dist = max(max_dist, dist_along_view)

    if min_dist == float('inf'):
        min_dist = 0.1
    else:
        min_dist = max(0.01, min_dist * (1 - margin))
    max_dist = max_dist * (1 + margin)

    camera_obj.data.clip_start = min_dist
    camera_obj.data.clip_end = max_dist

    print (f"Camera clipping adjusted: start={min_dist:.4f}, end={max_dist:.4f}")


# ===== Function: render_top_ortho_view_from_meshes =====
def render_top_ortho_view_from_meshes(output_path, margin=1, pixel_density=100, min_res=512, max_res=2048):
    """
    Renders a top-down orthographic view of all mesh objects in the current Blender scene.  
    The output image is automatically scaled to balance detail and performance, while ensuring  
    it stays within a given resolution range.

    Args:
        output_path (str): Path where the rendered image will be saved.  
        margin (float, optional): Scaling factor applied to the bounding box size (default=1).  
        pixel_density (int, optional): Pixels per unit in object space (default=100).  
        min_res (int, optional): Minimum resolution (applied to the smaller side) (default=512).  
        max_res (int, optional): Maximum resolution (applied to the larger side) (default=2048).  

    Returns:
        tuple:
            - int: Final image resolution along X.  
            - int: Final image resolution along Y.  
    """
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        return

    center, size, _ , _ = get_combined_bounding_box(meshes)
    size_x = size.x * margin
    size_y = size.y * margin

    res_x = size_x * pixel_density
    res_y = size_y * pixel_density

    aspect_ratio = res_x / res_y if res_y != 0 else 1.0

    scale_factor = 1.0
    if min(res_x, res_y) < min_res:
        scale_factor = min_res / min(res_x, res_y)
    elif max(res_x, res_y) > max_res:
        scale_factor = max_res / max(res_x, res_y)

    res_x = int(res_x * scale_factor)
    res_y = int(res_y * scale_factor)

    ortho_scale = max(size_x, size_y)

    scene = bpy.context.scene
    render = scene.render
    render.resolution_x = res_x
    render.resolution_y = res_y
    render.resolution_percentage = 100

    cam_data = bpy.data.cameras.new("OrthoCam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = ortho_scale

    cam = bpy.data.objects.new("OrthoCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000.0

    cam.location = center + mathutils.Vector((0, 0, 10))
    cam.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = cam

    render.filepath = output_path
    render.image_settings.file_format = 'PNG'
    render.film_transparent = True
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 16
    scene.cycles.preview_samples = 8
    scene.cycles.use_denoising = True

    bpy.ops.render.render(write_still=True)

    return res_x, res_y


# ===== Function: compute_camera_distance =====
def compute_camera_distance(fov_deg, obj_height, margin=1.2):
    """
    Returns the Z distance needed to see the whole object.
    """
    fov_rad = math.radians(fov_deg)
    return (obj_height * margin) / (2 * math.tan(fov_rad / 2))


# ===== Function: render_street_views_from_meshes =====
def render_street_views_from_meshes(output_dir, views=5, fov_deg=50.0, margin=1.2, altitudes=None):
    """
    Generates perspective "street view" renders around the meshes in the current Blender scene
    from multiple camera heights.
    
    Args:
        output_dir (str): Directory where the renders will be saved.
        views (int, optional): Number of camera views to generate around the model. Default = 5.
        fov_deg (float, optional): Camera field of view in degrees. Default = 50.0.
        margin (float, optional): Safety margin around the model to avoid cropping. Default = 1.2.
        altitudes (list[float], optional): List of relative heights (fractions of model height).
                                           Example [0.2, 0.5, 1.0] → 20%, 50% e 100% dell’altezza modello.
                                           If None, defaults to [0.5].
    """
    import bpy, math, os
    from mathutils import Vector

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == 'MESH']
    if not meshes:
        return

    center, size, _ , _ = get_combined_bounding_box(meshes)
    width, depth, height = size.x, size.y, size.z

    width_ratio = width / height if height > 0 else 1
    depth_ratio = depth / height if height > 0 else 1
    max_ratio = max(width_ratio, depth_ratio)

    zoom_factor = 1.0
    if max_ratio > 5:
        zoom_factor = 0.4
    elif max_ratio > 3:
        zoom_factor = 0.6
    elif max_ratio > 2:
        zoom_factor = 0.8

    target_size = max(width, depth)
    distance = compute_camera_distance(fov_deg, target_size, margin=margin) * zoom_factor

    cam_data = bpy.data.cameras.new("StreetCam")
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(fov_deg)

    cam = bpy.data.objects.new("StreetCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.image_settings.file_format = 'PNG'

    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000.0

    if altitudes is None:
        altitudes = [0.5]

    for h_idx, h_factor in enumerate(altitudes):
        z_cam = center.z + height * h_factor

        for i in range(views):
            angle = i * (2 * math.pi / views)
            x = center.x + distance * math.cos(angle)
            y = center.y + distance * math.sin(angle)
            z = z_cam

            cam.location = (x, y, z)
            direction = center - cam.location
            cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            output_path = os.path.join(output_dir, f"render_h{h_idx}_view{i}.png")
            scene.render.filepath = output_path
            bpy.ops.render.render(write_still=True)


# ===== Function: remove_cameras =====
def remove_cameras():
    """
    Removes all camera objects from the Blender scene, including any camera data blocks 
    that are no longer used. Ensures the scene is free of leftover cameras.
    """
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras:
        bpy.data.objects.remove(cam, do_unlink=True)
    
    for cam_data in bpy.data.cameras:
        if not cam_data.users:
            bpy.data.cameras.remove(cam_data)


# ===== Function: setup_hdri_lighting =====
def setup_hdri_lighting(hdri_path):
    """
    Sets up HDRI-based lighting in the Blender scene using the specified HDRI image.

    - Clears existing nodes in the world shader.
    - Creates an environment texture node, background shader, and world output node.
    - Connects the nodes to provide realistic lighting based on the HDRI image.

    Args:
        hdri_path (str): Path to the HDRI image file to use for lighting.

    Raises:
        FileNotFoundError: If the specified HDRI file does not exist.
    """
    if not os.path.isfile(hdri_path):
        raise FileNotFoundError(f"HDRI non trovato: {hdri_path}")

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in nodes:
        nodes.remove(node)

    node_background = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')

    links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    node_environment.image = bpy.data.images.load(hdri_path)


# ===== Function: ortho_camera_corners =====
def ortho_camera_corners(cam_obj, scene=None):
    """
    Computes the world-space coordinates of the four corners of an orthographic camera's view.

    Args:
        cam_obj (bpy.types.Object): The orthographic camera object.
        scene (bpy.types.Scene, optional): The Blender scene containing the camera. Defaults to the current scene.

    Returns:
        list[mathutils.Vector]: A list of four vectors representing the corners of the camera's orthographic frustum in world space.

    Raises:
        AssertionError: If the provided camera is not of type 'ORTHO'.
    """
    if scene is None:
        scene = bpy.context.scene
    
    render = scene.render
    cam_data = cam_obj.data
    
    assert cam_data.type == 'ORTHO', "Camera must be ortho"
    
    aspect_ratio = render.resolution_x / render.resolution_y
    height = cam_data.ortho_scale
    width = cam_data.ortho_scale * aspect_ratio

    corners_cam = [
        mathutils.Vector((-width/2, -height/2, 0)),
        mathutils.Vector(( width/2, -height/2, 0)),
        mathutils.Vector(( width/2,  height/2, 0)),
        mathutils.Vector((-width/2,  height/2, 0)),
    ]
    mat = cam_obj.matrix_world
    corners_world = [mat @ c for c in corners_cam]
    
    return corners_world


# ===== Function: postprocess_model =====
def postprocess_model(output_path, filename, corners_world, target_x):
    """
    Post-processes all mesh objects in the Blender scene by translating, scaling, and exporting them.

    - Translates all meshes so that the bottom-left corner aligns with the origin based on `corners_world`.
    - Scales all meshes uniformly along all axes to match the target width `target_x`.
    - Exports the processed meshes as a GLB file.

    Args:
        output_path (str): Path to save the exported GLB file.
        corners_world (list[mathutils.Vector]): World-space coordinates of reference corners for translation.
        target_x (float): Target size along the X-axis after scaling.
    """
    import mathutils
    
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    corner = max(corners_world, key=lambda v: (-v.x, v.y))

    # Calculate translation matrix
    translation_matrix = mathutils.Matrix.Translation((-corner[0], -corner[1], 0))
    
    # Apply translation to all meshes
    for obj in meshes:
        obj.data.transform(translation_matrix)
        obj.data.update()
    
    # Calculate scale factor
    _, _, bbox_max, bbox_min = get_combined_bounding_box(meshes)
    size_x = bbox_max[0] - bbox_min[0]
    scale_fact = target_x / size_x
    
    # Calculate scale matrix
    scale_matrix = mathutils.Matrix.Scale(scale_fact, 4)
    
    # Apply scale to all meshes
    for obj in meshes:
        obj.data.transform(scale_matrix)
        obj.data.update()
    
    # Calculate final transformation matrix (scale after translation)
    final_matrix = scale_matrix @ translation_matrix
    
    # Convert matrix to list for JSON serialization
    matrix_list = [list(row) for row in final_matrix]
    
    # Save matrix to JSON file
    json_path = os.path.join(output_path, "matrix_blender.json")
    with open(json_path, 'w') as f:
        json.dump(matrix_list, f, indent=2)
    
    # Export GLB
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(output_path, filename),
        export_format='GLB',
        use_selection=True,
        export_yup=True
    )
    

# ===== Function: get_ortho_camera_corners =====
def get_ortho_camera_corners():
    """
    Returns the world-space coordinates of the four corners of the active orthographic camera in the scene.

    Returns:
        list[mathutils.Vector]: A list of four vectors representing the corners of the orthographic camera's view.

    Raises:
        AssertionError: If the active camera is not orthographic.
    """
    scene = bpy.context.scene
    camera = scene.camera

    corners_cam = ortho_camera_corners(camera, scene=None)
    
    return corners_cam


def apply_transformations(obj):
    """
    Applies the active transformations (location, rotation, scale)
    to the specified object in Blender.

    Args:
        obj (bpy.types.Object): The object to which the transformations will be applied.
    """
    # Imposta l'oggetto come attivo
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Applica tutte le trasformazioni
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Deseleziona l'oggetto
    obj.select_set(False)


def check_point_cloud(obj):
    """
    Check if the given object is a point cloud (vertices without faces).
    If it is a point cloud, ensure it is visible in renders.

    Args:
        obj (bpy.types.Object): The object to check.
    
    Returns:
        bool: True if it's a point cloud, False otherwise.
    """
    if obj.type != 'MESH':
        return False
    
    mesh = obj.data
    is_point_cloud = len(mesh.polygons) == 0 and len(mesh.vertices) > 0

    if is_point_cloud:
        # Ensure object is visible in renders
        obj.hide_render = False
        
        # Optional: make points visible with small spheres (via geometry nodes or dupli verts)
        # For a quick visualization, you can enable 'Point Cloud' in viewport
        obj.display_type = 'WIRE'  # or 'BOUNDS', 'SOLID'


def get_world_matrix(obj):
    """
    Get the world matrix of a Blender mesh object.

    Args:
        obj (bpy.types.Object): The mesh object.
    
    Returns:
        mathutils.Matrix: The 4x4 world matrix of the object.
    """
    matrix = obj.matrix_world.copy()
    return matrix


# ===== Function: main =====
if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    output_folder = args.output_folder
    nr_view = args.streetviews

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    clear_scene()
    remove_cameras()

    setup_hdri_lighting(HDRI_PATH)

    obj = import_model(input_file)

    # Controlla se l'oggetto importato è una nuvola di punti - to do...
    check_point_cloud(obj)

    apply_transformations(obj)
    res_x, res_y = render_top_ortho_view_from_meshes(os.path.join(output_folder, 'top_view'))

    corners_world = get_ortho_camera_corners()

    remove_cameras()

    render_street_views_from_meshes(output_folder, views=nr_view, altitudes=[1.0,2.0])

    remove_cameras()

    postprocess_model(output_folder, base_name+"_scaled.glb", corners_world, res_x)
    clear_scene()