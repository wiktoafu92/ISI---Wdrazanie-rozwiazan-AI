from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from functools import lru_cache

from app.json_parser import parse_donut_xml
from app.models import Settings, DonutParsed


@lru_cache(maxsize=1)
def _load_donut_processor_and_model():
    settings = Settings()  # type: ignore
    if os.path.exists(settings.local_path_ocr) and os.path.isdir(settings.local_path_ocr):
        processor = DonutProcessor.from_pretrained(settings.local_path_ocr)
        model = VisionEncoderDecoderModel.from_pretrained(settings.local_path_ocr)
    else:
        processor = DonutProcessor.from_pretrained(settings.model_ocr)
        model = VisionEncoderDecoderModel.from_pretrained(settings.model_ocr)
        processor.save_pretrained(settings.local_path_ocr)
        model.save_pretrained(settings.local_path_ocr)

    return processor, model


def run_donut_ocr(image_path: str) -> DonutParsed:
    processor, model = _load_donut_processor_and_model()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = model.generate(pixel_values, max_length=512)  # type: ignore
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return parse_donut_xml(result)
