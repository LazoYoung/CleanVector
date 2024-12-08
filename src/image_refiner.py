import os

import matplotlib.pyplot as plt
import torch
from PIL import ImageDraw, ImageFont, Image
from transformers import CLIPProcessor, CLIPModel, DetrImageProcessor, DetrForObjectDetection

from src.saliency import ISNet, filter_images
from src.sentence_similarity import SentenceSimilarity
from src.util import read_images, random_path


class ImageRefiner:
    def __init__(self, prompt, output_dir):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.detr_proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.sentence_sim = SentenceSimilarity()
        self.prompt = prompt
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_clip_scores(self, images: any) -> list:
        with torch.no_grad():
            inputs = self.clip_proc(
                text=[self.prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # similarity score
        probs = logits_per_image.squeeze().softmax(dim=0).tolist()
        return probs

    def compute_detr_aabb(self, images: any, detr_threshold=0.9, prompt_threshold=0.3) -> list[dict]:
        """
        Find the most relevant object in each image.
        @param images: List of images or a single image
        @param detr_threshold: Minimum confidence required for an object detection
        @param prompt_threshold: Minimum semantic similarity required between an object and the prompt
        @return: list of each object
        """
        if not isinstance(images, list):
            images = [images]

        imsize = images[0].size

        with torch.no_grad():
            inputs = self.detr_proc(images=images, return_tensors="pt")
            outputs = self.detr_model(**inputs)

        target_sizes = torch.tensor([[imsize[1], imsize[0]] for _ in range(len(images))])
        results = self.detr_proc.post_process_object_detection(
            outputs=outputs,
            threshold=detr_threshold,
            target_sizes=target_sizes,
        )
        prompt_embed = self.sentence_sim.embedding(self.prompt)
        objects = []

        for result in results:
            img_objects = []
            max_similarity = 0
            relevant_label = None

            for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                score = score.item()
                label = self.detr_model.config.id2label[label.item()]
                similarity = self.sentence_sim.compute_score(prompt_embed, label)

                if similarity > max_similarity:
                    max_similarity = similarity
                    relevant_label = label

                box = [round(i, 2) for i in box.tolist()]
                img_objects.append({
                    'score': score,
                    'label': label,
                    'box': box,
                })
                # print(f"Detected {label} with confidence {score:.3f} at location {box}")

            # todo - DETR fails to meet threshold if prompt is too expressive
            # perhaps we could adjust threshold based on prompt length
            if max_similarity > prompt_threshold and relevant_label:
                img_objects = filter(lambda _obj: _obj['label'] == relevant_label, img_objects)
                obj = sorted(img_objects, key=lambda _obj: _obj['score'])[-1]
                objects.append(obj)
                print(f"Found {obj['label']} with confidence {obj['score']} and similarity {max_similarity} at location {obj['box']}")
            else:
                print(f"No object found. Max similarity was {max_similarity} with label {relevant_label}")
                objects.append(None)

        return objects

    def detr_label(self, label_emb):
        return self.detr_model.config.id2label[label_emb.item()]

    @staticmethod
    def visualize(images, clip_scores, detr_results=None):
        if detr_results is None:
            detr_results = [None] * len(clip_scores)
        for idx, (score, obj) in enumerate(zip(clip_scores, detr_results)):
            image = images[idx]
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=24)

            if obj:
                obj_score = obj['score']
                label = obj['label']
                box = [round(i, 2) for i in obj['box']]
                text = f"{label}: {obj_score:.3f}"
                x1, y1, x2, y2 = tuple(box)
                draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
                draw.text((x1, y1), text=text, fill='white', font=font)

            plt.suptitle(f"CLIP score: {score:.3f}")
            plt.imshow(image)
            plt.show()

    def augment(self, images, detr_results=None, top_k=7):
        cropped = []

        if detr_results is None:
            cropped = images
        else:
            # crop images according to DETR AABB
            for idx, obj in enumerate(detr_results):
                if obj:
                    image = images[idx]
                    image = image.crop(obj['box'])
                    cropped.append(image)
            if len(cropped) == 0:
                print("No image were sufficient")
                return

        # remove background
        images = cropped
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_net = ISNet(device)
        saliency_maps = is_net.segment(images)
        images = filter_images(images, saliency_maps)
        images = [Image.fromarray(image) for image in images]

        # compute semantic similarity
        scores = self.compute_clip_scores(images)
        scores = scores if isinstance(scores, list) else [scores]
        images = [(score, image) for score, image in zip(scores, images)]
        images = sorted(images, key=lambda x: x[0], reverse=True)[:top_k]
        images = [image for score, image in images]

        for image, score in zip(images, scores):
            plt.imshow(image)
            plt.suptitle(f"CLIP score: {score:.3f}")
            plt.show()
            image.save(random_path(ext="png", dir=self.output_dir))


def refine(images, prompt, output_dir, detect_objects=True, prompt_threshold=0.3):
    if len(images) == 0:
        raise ValueError("Image should not be empty.")
    refiner = ImageRefiner(prompt, output_dir)
    clip_scores = refiner.compute_clip_scores(images)
    detr_results = refiner.compute_detr_aabb(images, prompt_threshold) if detect_objects else None
    # refiner.visualize(images, clip_scores, detr_results)
    refiner.augment(images, detr_results)


if __name__ == "__main__":
    dir = "../output/diffused"
    prompt = "stop sign"
    refine(read_images(dir), prompt, dir)
