import os

import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import CLIPProcessor, CLIPModel, DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt


class ImageScoreComputer:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.detr_proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def compute_scores(self, images, token: str):
        with torch.no_grad():
            inputs = self.clip_proc(
                text=[token],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # similarity score
        print(logits_per_image, logits_per_image.shape)
        probs = logits_per_image.squeeze().softmax(dim=0).tolist()
        return probs

    def detect_objects(self, images, token):
        imsize = images[0].size
        with torch.no_grad():
            inputs = self.detr_proc(images=images, return_tensors="pt")
            outputs = self.detr_model(**inputs)
        target_sizes = torch.tensor([[imsize[1], imsize[0]] for _ in range(len(images))])
        results = self.detr_proc.post_process_object_detection(
            outputs=outputs,
            threshold=0.3,
            target_sizes=target_sizes,
        )
        for result in results:
            for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    "Detected %s with confidence %.3f at location %s" %
                    (
                        self.detr_model.config.id2label[label.item()],
                        round(score.item(), 3),
                        box,
                    )
                )
        return results

    def detr_label(self, label):
        return self.detr_model.config.id2label[label.item()]


def main():
    computer = ImageScoreComputer()
    dir = "../output/diffused"
    token = "stop sign"
    images = []

    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)

        if os.path.isfile(path) and filename.split(".")[-1].lower() == "png":
            try:
                with Image.open(path) as img:
                    img.load()
                    images.append(img)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    scores = computer.compute_scores(images, token)
    results = computer.detect_objects(images, token)

    for idx, (score, result) in enumerate(zip(scores, results)):
        image = images[idx]
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=24)

        for detr_score, label, box in zip(result['scores'], result['labels'], result['boxes']):
            box = [round(i, 2) for i in box.tolist()]
            text = f"{computer.detr_label(label)}: {detr_score:.3f}"
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
            draw.text((x1, y1), text=text, fill='white', font=font)

        plt.suptitle(f"CLIP score: {score:.3f}")
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    main()
