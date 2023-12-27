# idea from <https://huggingface.co/spaces/toshas/marigold>

import time
import tempfile
from PIL import Image, ImageOps, ImageFilter
import gradio as gr
from modules import shared, script_callbacks
from modules.control.processors import Processor
import extrude # pylint: disable=wrong-import-order


gr_height = 512
MODELS=['None', 'Midas Depth Hybrid', 'Leres Depth', 'Zoe Depth', 'Normal Bae']
processor: Processor = None
css = """
    #depth_3d_images { gap: 0; }
"""


def process_image(
            model: str,
            input_image: Image.Image,
            invert,
            filter_size,
            colors,
            lights,
            scale,
            plane_near,
            plane_back,
            emboss,
            f_thic,
            f_near,
            f_back,
):
    global processor # pylint: disable=global-statement
    if input_image is None:
        return None, None, None
    if model == 'None':
        processor = None
    elif processor is None or processor.processor_id != model:
        processor = Processor(model)
    t0 = time.time()
    if processor is None:
        output_image = input_image
    else: # get depth image
        output_image = processor(input_image)
    t1 = time.time()
    shared.log.info(f'Depth3D: process=depth model={model} time={t1-t0:.3f}')

    # postprocess depth image
    if invert:
        output_image = ImageOps.invert(output_image)
    output_image = output_image.convert("L")
    if filter_size > 0:
        output_image = output_image.filter(ImageFilter.MedianFilter(size=2 * filter_size - 1))

    # get 3d model from depth image
    extrude_dict = {
        'input_image': input_image,
        'depth_image': output_image,
        'output_model_scale': scale,
        'coef_near': plane_near,
        'coef_far': plane_back,
        'emboss': emboss,
        'f_thic': f_thic,
        'f_near': f_near,
        'f_back': f_back,
        'vertex_colors': colors,
        'scene_lights': lights,
        'path_glb': tempfile.NamedTemporaryFile(delete=False, suffix='.glb', dir=shared.opts.temp_dir).name,
        'path_stl': tempfile.NamedTemporaryFile(delete=False, suffix='.stl', dir=shared.opts.temp_dir).name,
    }
    shared.log.debug(f'Depth3D args: {extrude_dict}')
    path_glb, path_stl = extrude.extrude_depth_3d(**extrude_dict)
    t2 = time.time()
    shared.log.info(f'Depth3D: process=extrude files=[{path_glb}, {path_stl}] time={t2-t1:.3f}')
    return output_image, path_glb, [path_glb, path_stl]


def create_ui(_blocks: gr.Blocks = None):
    with gr.Blocks(analytics_enabled = False, css=css) as depth_ui:
        with gr.Row(elem_id = 'depth_3d_images'):
            with gr.Column():
                input_image = gr.Image(label="Input", show_label=True, type="pil", source="upload", interactive=True, tool="editor", height=gr_height)
            with gr.Column():
                output_image = gr.Image(label="Depth", show_label=True, type="pil", interactive=False, tool="editor", height=gr_height)
            with gr.Column():
                depth_image = gr.Model3D(label="Viewport", show_label=True, camera_position=(75.0, 90.0, 75.0), zoom_speed=1.0, height=gr_height)

        with gr.Row(elem_id = 'depth_3d_settings'):
            with gr.Accordion(open=True, label="Depth", elem_id="control_input", elem_classes=["small-accordion"]):
                model = gr.Dropdown(label="Model", choices=MODELS, value="None")
                invert = gr.Checkbox(label="Invert", value=True)
                filter_size = gr.Slider(label="Filter", minimum=0, maximum=10, step=1, value=3)
            with gr.Accordion(open=True, label="Extrude", elem_id="control_input", elem_classes=["small-accordion"]):
                with gr.Row():
                    colors = gr.Checkbox(label="Colors", value=True)
                    lights = gr.Checkbox(label="Lights", value=True)
                with gr.Row():
                    scale = gr.Slider(label="Scale", minimum=1, maximum=1000, step=1, value=100)
                    emboss = gr.Slider(label="Emboss", minimum=0, maximum=1, step=0.05, value=0.2)
                with gr.Row():
                    plane_near = gr.Slider(label="Near plane", minimum=0, maximum=1, step=0.05, value=0)
                    plane_back = gr.Slider(label="Back plane", minimum=0, maximum=1, step=0.05, value=1)
                with gr.Row():
                    f_thic = gr.Slider(label="Frame thickness", minimum=0, maximum=1, step=0.05, value=0.5, visible=False)
                    f_near = gr.Slider(label="Frame near", minimum=0, maximum=1, step=0.05, value=0.5)
                    f_back = gr.Slider(label="Frame back", minimum=0, maximum=1, step=0.05, value=0.5)
            with gr.Accordion(open=True, label="Camera", elem_id="control_input", elem_classes=["small-accordion"], visible=False):
                alpha = gr.Slider(label="Alpha", minimum=0, maximum=360, step=1, value=75)
                beta = gr.Slider(label="Alpha", minimum=0, maximum=360, step=1, value=90)
                radius = gr.Slider(label="Radius", minimum=0, maximum=100, step=1, value=10)
                alpha.change(fn=lambda a,b,r: gr.update(camera_position=(a,b,r)), inputs=[alpha, beta, radius], outputs=[depth_image])
                beta.change(fn=lambda a,b,r: gr.update(camera_position=(a,b,r)), inputs=[alpha, beta, radius], outputs=[depth_image])
                radius.change(fn=lambda a,b,r: gr.update(camera_position=(a,b,r)), inputs=[alpha, beta, radius], outputs=[depth_image])

        with gr.Row(elem_id = 'depth_3d_files'):
            files = gr.Files(label="Files", elem_id="3d_depth_download", interactive=False)

        inputs = [
            model,
            input_image,
            invert,
            filter_size,
            colors,
            lights,
            scale,
            plane_near,
            plane_back,
            emboss,
            f_thic,
            f_near,
            f_back,
        ]
        for ctrl in inputs:
            if hasattr(ctrl, 'change'):
                ctrl.change(fn=process_image, inputs=inputs, outputs=[output_image, depth_image, files])

    return [(depth_ui, 'Depth 3D', 'depth_3d')]


script_callbacks.on_ui_tabs(create_ui)

"""
    update(['Midas Depth Hybrid', 'params', 'bg_th'], settings[1])
    update(['Midas Depth Hybrid', 'params', 'depth_and_normal'], settings[2])
    update(['Leres Depth', 'params', 'boost'], settings[11])
    update(['Leres Depth', 'params', 'thr_a'], settings[12])
    update(['Leres Depth', 'params', 'thr_b'], settings[13])
    update(['Zoe Depth', 'params', 'gamma_corrected'], settings[23])
"""
