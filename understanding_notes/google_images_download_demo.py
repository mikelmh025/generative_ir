# Using the Google Images Download library
# First install it with: pip install google_images_download

from google_images_download import google_images_download

# Define the directories first
save_dir = "downloaded_images"  # Main output directory
image_directory = "cat_photos"  # Subdirectory for the specific search

# Create an instance of the google_images_download class
response = google_images_download.googleimagesdownload()

# Define search parameters
prompt = "a photo of a cat"
arguments = {
    "keywords": prompt,
    "limit": 5,
    "print_urls": True,
    "output_directory": save_dir,
    "image_directory": image_directory,
    "format": "jpg",
    "silent_mode": False
}

# Execute the download
paths = response.download(arguments)

# Print the paths to verify downloads
print(f"Images downloaded to: {paths}")

