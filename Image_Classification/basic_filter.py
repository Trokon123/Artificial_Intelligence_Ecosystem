"""
Image filters.

Part 2 of the Image Classification project.
  blur   - the original starter filter (Gaussian blur)
  comic  - my custom filter: turns a photo into a comic-book / pop-art panel

Run:  python3 basic_filter.py
Then type an image filename and choose a filter.
"""

import os

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


def apply_blur_filter(image_path, output_path="blurred_image.png"):
    """Starter filter: shrink the image to 128x128 and apply a Gaussian blur."""
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        # Each pixel becomes a weighted average of its neighbours within radius 2
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imshow(img_blurred)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_comic_filter(image_path, output_path="comic_image.png",
                       color_levels=3, saturation=1.8, contrast=1.3,
                       edge_threshold=40, max_size=640):
    """Custom filter: make a photo look like a comic-book panel.

    Three ideas are combined:
      1. Flat colour  - smooth away fine texture, then reduce the number of
                        colours (posterize) so areas become solid blocks.
      2. Bold colour  - raise saturation and contrast so it looks printed.
      3. Ink outlines - detect edges and draw them in black on top.

    Parameters you can tweak:
      color_levels   bits kept per colour channel (2 = very flat, 4 = subtle)
      saturation     1.0 = unchanged, higher = more vivid
      contrast       1.0 = unchanged, higher = punchier
      edge_threshold lower = more outlines, higher = only the strongest edges
      max_size       longest side of the output in pixels
    """
    try:
        img = Image.open(image_path).convert("RGB")
        # Shrink large photos (keeps the aspect ratio, unlike the blur filter)
        img.thumbnail((max_size, max_size))

        # --- 1. Flat colour -------------------------------------------------
        # A median filter removes small detail but keeps edges sharp
        smooth = img.filter(ImageFilter.MedianFilter(size=5))

        # --- 2. Bold colour -------------------------------------------------
        vivid = ImageEnhance.Color(smooth).enhance(saturation)
        vivid = ImageEnhance.Contrast(vivid).enhance(contrast)
        # Keep only the top `color_levels` bits of each channel
        flat = ImageOps.posterize(vivid, color_levels)

        # --- 3. Ink outlines ------------------------------------------------
        gray = ImageOps.grayscale(img).filter(ImageFilter.GaussianBlur(radius=1))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        # Strong edge -> black (0), everything else -> white (255)
        ink = edges.point(lambda p: 0 if p > edge_threshold else 255)
        # Thicken the black lines a little (MinFilter spreads dark pixels)
        ink = ink.filter(ImageFilter.MinFilter(size=3))
        # FIND_EDGES leaves a false line around the border, so paint it white
        ink = ImageOps.expand(ImageOps.crop(ink, border=2), border=2, fill=255)

        # Multiply: white ink leaves the colour alone, black ink turns it black
        comic = ImageChops.multiply(flat, ink.convert("RGB"))

        comic.save(output_path)
        print(f"Comic image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Image Processor (type 'exit' to quit)")
    print("Available filters: blur, comic\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        choice = input("Choose filter ('blur' or 'comic'): ").strip().lower()
        # Build the output name from the input name, e.g. dog.jpg -> dog_comic.jpg
        base, ext = os.path.splitext(image_path)

        if choice == "blur":
            apply_blur_filter(image_path, f"{base}_blurred{ext}")
        elif choice == "comic":
            apply_comic_filter(image_path, f"{base}_comic{ext}")
        else:
            print("Unknown filter. Please type 'blur' or 'comic'.")
