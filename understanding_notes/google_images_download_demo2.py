# import os
# import re
# import json
# import time
# import argparse
# import urllib.request
# import urllib.parse
# import urllib.error
# import ssl
# from pathlib import Path
# from bs4 import BeautifulSoup
# import random
# import sys

# class GoogleImageDownloader:
#     def __init__(self):
#         self.ssl_context = ssl._create_unverified_context()
#         self.url_pattern = re.compile(r'(?:http|https)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#             'Accept-Language': 'en-US,en;q=0.5',
#             'Connection': 'keep-alive',
#             'Upgrade-Insecure-Requests': '1',
#         }
#         # User agents list for rotation
#         self.user_agents = [
#             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
#             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
#             'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0',
#             'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
#         ]

#     def download(self, arguments):
#         """
#         Main function to handle the download process
#         """
#         keywords = arguments["keywords"]
#         keywords = [str(keyword).strip() for keyword in keywords.split(',')]
        
#         # Create directories
#         main_directory = arguments.get('output_directory', 'downloads')
        
#         # Create main directory if it doesn't exist
#         if not os.path.exists(main_directory):
#             os.makedirs(main_directory)
            
#         downloaded_image_paths = []
        
#         # Process each keyword
#         for keyword in keywords:
#             print(f"\nSearching for: {keyword}")
            
#             # Create keyword directory
#             image_directory = arguments.get('image_directory')
#             if not image_directory:
#                 image_directory = keyword.replace(" ", "_")
            
#             dir_name = os.path.join(main_directory, image_directory)
            
#             # Create the directory
#             if not os.path.exists(dir_name):
#                 os.makedirs(dir_name)
                
#             print(f"Saving images to: {dir_name}")
            
#             # Build the Google search URL
#             search_url = self.build_search_url(keyword, arguments)
            
#             # Download the page
#             html = self.download_page(search_url)
#             if html:
#                 # Extract and download images
#                 image_urls = self.extract_image_urls(html, arguments['limit'])
                
#                 if not image_urls:
#                     print(f"Couldn't find any images for: {keyword}")
#                 else:
#                     print(f"Found {len(image_urls)} images")
                    
#                     # Download each image
#                     count = 1
#                     for image_url in image_urls:
#                         if count > arguments['limit']:
#                             break
                            
#                         if arguments.get('print_urls', False):
#                             print(f"\nImage URL: {image_url}")
                            
#                         try:
#                             # Extract extension from URL or default to jpg
#                             extension = self.get_extension_from_url(image_url, arguments.get('format', 'jpg'))
                            
#                             # Create filename
#                             filename = self.get_filename(keyword, count, arguments, extension)
                            
#                             # Full path to save image
#                             image_path = os.path.join(dir_name, filename)
                            
#                             # Download the image
#                             success = self.save_image(image_url, image_path, arguments)
                            
#                             if success:
#                                 if not arguments.get('silent_mode', False):
#                                     print(f"Downloaded {count}/{arguments['limit']}: {filename}")
#                                 downloaded_image_paths.append(image_path)
#                                 count += 1
#                             else:
#                                 print(f"Failed to download: {image_url}")
                                
#                             # Apply delay if specified
#                             if arguments.get('delay'):
#                                 time.sleep(int(arguments['delay']))
                                
#                         except Exception as e:
#                             print(f"ERROR: {e}")
#                             continue
#             else:
#                 print(f"Failed to download search page for: {keyword}")
                
#         return downloaded_image_paths
    
#     def build_search_url(self, keyword, arguments):
#         """
#         Build the Google Images search URL
#         """
#         keyword = keyword.replace(' ', '+')
#         url = f"https://www.google.com/search?q={keyword}&tbm=isch"
        
#         # Add additional search parameters
#         if arguments.get('size'):
#             size_map = {
#                 'large': 'isz:l',
#                 'medium': 'isz:m',
#                 'icon': 'isz:i'
#             }
#             if arguments['size'] in size_map:
#                 url += f"&tbs={size_map[arguments['size']]}"
                
#         if arguments.get('color'):
#             color_map = {
#                 'red': 'ic:specific,isc:red',
#                 'orange': 'ic:specific,isc:orange',
#                 'yellow': 'ic:specific,isc:yellow',
#                 'green': 'ic:specific,isc:green',
#                 'blue': 'ic:specific,isc:blue',
#                 'purple': 'ic:specific,isc:purple',
#                 'pink': 'ic:specific,isc:pink',
#                 'white': 'ic:specific,isc:white',
#                 'gray': 'ic:specific,isc:gray',
#                 'black': 'ic:specific,isc:black',
#                 'brown': 'ic:specific,isc:brown'
#             }
#             if arguments['color'] in color_map:
#                 url += f"&tbs={color_map[arguments['color']]}"
                
#         if arguments.get('type'):
#             type_map = {
#                 'face': 'itp:face',
#                 'photo': 'itp:photo',
#                 'clipart': 'itp:clipart',
#                 'lineart': 'itp:lineart',
#                 'animated': 'itp:animated'
#             }
#             if arguments['type'] in type_map:
#                 url += f"&tbs={type_map[arguments['type']]}"
                
#         return url
    
#     def download_page(self, url):
#         """
#         Download the search results page
#         """
#         try:
#             # Rotate user agents
#             self.headers['User-Agent'] = random.choice(self.user_agents)
            
#             # Create a request object with headers
#             request = urllib.request.Request(url, headers=self.headers)
            
#             # Open URL and read response
#             response = urllib.request.urlopen(request, context=self.ssl_context)
            
#             # Check if response is valid
#             if response.getcode() != 200:
#                 print(f"Error: Got status code {response.getcode()}")
#                 return None
                
#             # Read and decode the page content
#             html = response.read().decode('utf-8')
#             return html
            
#         except Exception as e:
#             print(f"ERROR: {e}")
#             return None
    
#     def extract_image_urls(self, html, limit):
#         """
#         Extract image URLs from the HTML
#         """
#         image_urls = []
        
#         # Try BeautifulSoup parsing first
#         soup = BeautifulSoup(html, 'html.parser')
        
#         # Look for "DS1iW" class which contains the image
#         img_elements = soup.find_all('img', class_='DS1iW')
        
#         if img_elements:
#             for img in img_elements:
#                 if len(image_urls) >= limit:
#                     break
                    
#                 img_url = img.get('src')
#                 if img_url and self.is_valid_url(img_url):
#                     image_urls.append(img_url)
        
#         # If we didn't find enough images, try regex approach
#         if len(image_urls) < limit:
#             # Find all src attributes
#             src_pattern = re.compile(r'src="(https://[^"]+)"')
#             matches = src_pattern.findall(html)
            
#             # Filter for image URLs
#             for match in matches:
#                 if len(image_urls) >= limit:
#                     break
                
#                 if self.is_valid_url(match) and any(ext in match.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
#                     if match not in image_urls:
#                         image_urls.append(match)
        
#         # Third approach: look for "data:" attributes if we still need more images
#         if len(image_urls) < limit:
#             data_pattern = re.compile(r'data:image/([^;]+);base64,([^"]+)')
#             data_matches = data_pattern.findall(html)
            
#             # Process data URI images (we'll skip these for now as they require special handling)
#             # This would need to be implemented if data URI images are needed
        
#         return image_urls
    
#     def is_valid_url(self, url):
#         """
#         Check if a URL is valid
#         """
#         return bool(self.url_pattern.match(url))
    
#     def get_extension_from_url(self, url, default_extension='jpg'):
#         """
#         Extract file extension from URL or use default
#         """
#         if url.lower().endswith('.jpg') or url.lower().endswith('.jpeg'):
#             return 'jpg'
#         elif url.lower().endswith('.png'):
#             return 'png'
#         elif url.lower().endswith('.gif'):
#             return 'gif'
#         elif url.lower().endswith('.webp'):
#             return 'webp'
#         else:
#             return default_extension
    
#     def get_filename(self, keyword, count, arguments, extension):
#         """
#         Generate filename for the image
#         """
#         keyword = keyword.replace(" ", "_")
        
#         if arguments.get('no_numbering', False):
#             filename = f"{keyword}.{extension}"
#         else:
#             filename = f"{keyword}_{count}.{extension}"
            
#         if arguments.get('prefix'):
#             filename = f"{arguments['prefix']}{filename}"
            
#         return filename
    
#     def save_image(self, url, path, arguments):
#         """
#         Download and save the image
#         """
#         if arguments.get('no_download', False):
#             return True
            
#         try:
#             # Rotate user agents
#             self.headers['User-Agent'] = random.choice(self.user_agents)
            
#             # Create request
#             request = urllib.request.Request(url, headers=self.headers)
            
#             # Open image URL
#             with urllib.request.urlopen(request, context=self.ssl_context) as response:
#                 # Check if response is valid
#                 if response.getcode() != 200:
#                     print(f"Failed to download image (status code: {response.getcode()})")
#                     return False
                
#                 # Get image data
#                 image_data = response.read()
                
#                 # Get file size
#                 file_size = len(image_data)
                
#                 # Print file size if requested
#                 if arguments.get('print_size', False):
#                     print(f"Image size: {self.format_size(file_size)}")
                
#                 # Save the image
#                 with open(path, 'wb') as f:
#                     f.write(image_data)
                    
#                 return True
                
#         except Exception as e:
#             print(f"ERROR downloading {url}: {e}")
#             return False
    
#     def format_size(self, size_bytes):
#         """
#         Format bytes to human-readable size
#         """
#         if size_bytes < 1024:
#             return f"{size_bytes} bytes"
#         elif size_bytes < 1024 * 1024:
#             return f"{size_bytes / 1024:.1f} KB"
#         else:
#             return f"{size_bytes / (1024 * 1024):.1f} MB"

# # Main implementation for the command-line tool
# def main():
#     parser = argparse.ArgumentParser(description='Download images from Google Images')
#     parser.add_argument('-k', '--keywords', help='Keywords to search for', default="a photo of a cat")
#     parser.add_argument('-l', '--limit', help='Maximum number of images to download', type=int, default=10)
#     parser.add_argument('-o', '--output_directory', help='Main directory to save images', default='downloads')
#     parser.add_argument('-i', '--image_directory', help='Sub-directory for the images')
#     parser.add_argument('-f', '--format', help='Image format to save as', default='jpg')
#     parser.add_argument('-s', '--size', help='Image size (large, medium, icon)')
#     parser.add_argument('-c', '--color', help='Filter by color (red, orange, yellow, green, blue, purple, pink, white, gray, black, brown)')
#     parser.add_argument('-t', '--type', help='Image type (face, photo, clipart, lineart, animated)')
#     parser.add_argument('-d', '--delay', help='Delay between downloads in seconds', type=int)
#     parser.add_argument('--print_urls', help='Print image URLs', action='store_true')
#     parser.add_argument('--print_size', help='Print image sizes', action='store_true')
#     parser.add_argument('--no_download', help='Skip actual downloading', action='store_true')
#     parser.add_argument('--no_numbering', help='Do not add numbers to filenames', action='store_true')
#     parser.add_argument('--prefix', help='Add prefix to filenames')
#     parser.add_argument('--silent_mode', help='Suppress output', action='store_true')
    
#     args = parser.parse_args()
    
#     # Convert args to dict
#     arguments = vars(args)
    
#     # Download images
#     downloader = GoogleImageDownloader()
#     image_paths = downloader.download(arguments)
    
#     # Print results
#     if image_paths:
#         print(f"\nSuccessfully downloaded {len(image_paths)} images:")
#         for path in image_paths:
#             print(f"  - {path}")
#     else:
#         print("\nNo images were downloaded.")

# # Example usage as a module
# class googleimagesdownload:
#     def __init__(self):
#         self.downloader = GoogleImageDownloader()
        
#     def download(self, arguments):
#         """
#         Download images based on the provided arguments
#         """
#         return self.downloader.download(arguments)

# # Entry point
# if __name__ == "__main__":
#     main()


import os
import re
import json
import time
import random
import argparse
import urllib.request
import urllib.parse
import urllib.error
import ssl
from pathlib import Path
from bs4 import BeautifulSoup
import logging

class GoogleImageCrawler:
    def __init__(self, debug=False):
        # Setup logging
        self.logger = logging.getLogger('GoogleImageCrawler')
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # SSL context for requests
        self.ssl_context = ssl._create_unverified_context()
        
        # URL and image patterns
        self.url_pattern = re.compile(r'(?:http|https)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.image_patterns = [
            re.compile(r'<img[^>]*src="([^"]*)"[^>]*class="[^"]*DS1iW[^"]*"'),  # Modern class pattern
            re.compile(r'<img[^>]*class="[^"]*DS1iW[^"]*"[^>]*src="([^"]*)"'),  # Alternative order
            re.compile(r'<img[^>]*src="(https://encrypted-tbn0[^"]*)"'),        # Encrypted image URLs
            re.compile(r'imgurl=(https?://[^&]+)&'),                            # URL parameter pattern
            re.compile(r'"ou":"(https?://[^"]+)"'),                             # JSON pattern
            re.compile(r'<img[^>]*src="([^"]*)"[^>]*>'),                        # Generic img pattern (last resort)
        ]
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
        ]
        
        # Request headers
        self.headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'TE': 'Trailers',
        }

    def download(self, arguments):
        """
        Main function to handle the download process
        """
        keywords = arguments["keywords"]
        keywords = [str(keyword).strip() for keyword in keywords.split(',')]
        limit = int(arguments.get('limit', 10))
        
        # Create directories
        main_directory = arguments.get('output_directory', 'downloads')
        
        # Create main directory if it doesn't exist
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)
            
        downloaded_image_paths = []
        
        # Process each keyword
        for keyword in keywords:
            self.logger.info(f"Searching for: {keyword}")
            
            # Create keyword directory
            image_directory = arguments.get('image_directory')
            if not image_directory:
                image_directory = keyword.replace(" ", "_")
            
            dir_name = os.path.join(main_directory, image_directory)
            
            # Create the directory
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            self.logger.info(f"Saving images to: {dir_name}")
            
            # Build the Google search URL
            search_url = self.build_search_url(keyword, arguments)
            
            # Download the page
            html = self.download_page(search_url)
            if html:
                # Extract and download images
                image_urls = self.extract_image_urls(html, limit)
                
                if not image_urls:
                    self.logger.warning(f"Couldn't find any images for: {keyword}")
                else:
                    self.logger.info(f"Found {len(image_urls)} images")
                    
                    # Download each image
                    count = 1
                    for image_url in image_urls:
                        if count > limit:
                            break
                            
                        if arguments.get('print_urls', False):
                            self.logger.info(f"Image URL: {image_url}")
                            
                        try:
                            # Extract extension from URL or default to jpg
                            extension = self.get_extension_from_url(image_url, arguments.get('format', 'jpg'))
                            
                            # Create filename
                            filename = self.get_filename(keyword, count, arguments, extension)
                            
                            # Full path to save image
                            image_path = os.path.join(dir_name, filename)
                            
                            # Download the image
                            success = self.save_image(image_url, image_path, arguments)
                            
                            if success:
                                if not arguments.get('silent_mode', False):
                                    self.logger.info(f"Downloaded {count}/{limit}: {filename}")
                                downloaded_image_paths.append(image_path)
                                count += 1
                            else:
                                self.logger.warning(f"Failed to download: {image_url}")
                                
                            # Apply delay if specified
                            if arguments.get('delay'):
                                time.sleep(int(arguments['delay']))
                                
                        except Exception as e:
                            self.logger.error(f"Error downloading {image_url}: {e}")
                            continue
            else:
                self.logger.error(f"Failed to download search page for: {keyword}")
                
        return downloaded_image_paths
    
    def build_search_url(self, keyword, arguments):
        """
        Build the Google Images search URL
        """
        keyword = keyword.replace(' ', '+')
        url = f"https://www.google.com/search?q={keyword}&tbm=isch"
        
        # Add additional search parameters
        if arguments.get('size'):
            size_map = {
                'large': 'isz:l',
                'medium': 'isz:m',
                'icon': 'isz:i'
            }
            if arguments['size'] in size_map:
                url += f"&tbs={size_map[arguments['size']]}"
                
        if arguments.get('color'):
            color_map = {
                'red': 'ic:specific,isc:red',
                'orange': 'ic:specific,isc:orange',
                'yellow': 'ic:specific,isc:yellow',
                'green': 'ic:specific,isc:green',
                'blue': 'ic:specific,isc:blue',
                'purple': 'ic:specific,isc:purple',
                'pink': 'ic:specific,isc:pink',
                'white': 'ic:specific,isc:white',
                'gray': 'ic:specific,isc:gray',
                'black': 'ic:specific,isc:black',
                'brown': 'ic:specific,isc:brown'
            }
            if arguments['color'] in color_map:
                url += f"&tbs={color_map[arguments['color']]}"
                
        if arguments.get('type'):
            type_map = {
                'face': 'itp:face',
                'photo': 'itp:photo',
                'clipart': 'itp:clipart',
                'lineart': 'itp:lineart',
                'animated': 'itp:animated'
            }
            if arguments['type'] in type_map:
                url += f"&tbs={type_map[arguments['type']]}"
        
        return url
    
    def download_page(self, url):
        """
        Download the search results page
        """
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read())
        
        return respData
        # try:
        #     # Rotate user agents
        #     self.headers['User-Agent'] = random.choice(self.user_agents)
            
        #     # Create a request object with headers
        #     request = urllib.request.Request(url, headers=self.headers)
            
        #     # Open URL and read response
        #     with urllib.request.urlopen(request, context=self.ssl_context, timeout=30) as response:
        #         # Check if response is valid
        #         if response.getcode() != 200:
        #             self.logger.error(f"Got status code {response.getcode()}")
        #             return None
                    
        #         # Read and decode the page content
        #         html = response.read().decode('utf-8')
        #         self.logger.debug(f"Downloaded page with size: {len(html)} bytes")
        #         return html
                
        # except Exception as e:
        #     self.logger.error(f"Error downloading page: {e}")
        #     return None
    
    def extract_image_urls(self, html, limit):
        """
        Extract image URLs from the HTML using multiple methods
        """
        image_urls = []
        original_urls = set()  # To track unique URLs
        
        # Method 1: Try each regex pattern
        for pattern in self.image_patterns:
            matches = pattern.findall(html)
            for match in matches:
                if len(image_urls) >= limit * 2:  # Get more than needed in case some fail
                    break
                
                if match and self.is_valid_url(match):
                    # Skip data URLs
                    if match.startswith('data:'):
                        continue
                    
                    # Skip very small thumbnails
                    if 's1-w' in match and 's1-h' in match:
                        sizes = re.findall(r's1-w(\d+)-h(\d+)', match)
                        if sizes and int(sizes[0][0]) < 50 and int(sizes[0][1]) < 50:
                            continue
                    
                    # Clean up URL
                    clean_url = self.clean_url(match)
                    if clean_url and clean_url not in original_urls:
                        image_urls.append(clean_url)
                        original_urls.add(clean_url)
        
        # Method 2: Try to find JSON data in script tags
        json_data_pattern = re.compile(r'AF_initDataCallback\((.*?)\);')
        script_tags = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
        for script in script_tags:
            json_matches = json_data_pattern.findall(script)
            for json_match in json_matches:
                try:
                    # Extract URLs from JSON data
                    urls = re.findall(r'"(https://[^"]+\.(?:jpg|jpeg|png|gif|webp))"', json_match)
                    for url in urls:
                        if len(image_urls) >= limit * 2:
                            break
                        
                        if url and url not in original_urls:
                            image_urls.append(url)
                            original_urls.add(url)
                except:
                    pass
        
        # Method 3: Use BeautifulSoup as a last resort
        if len(image_urls) < limit:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find all img tags
                img_tags = soup.find_all('img')
                for img in img_tags:
                    if len(image_urls) >= limit * 2:
                        break
                    
                    # Try different attributes where URLs might be found
                    for attr in ['src', 'data-src', 'data-url', 'data-srcset']:
                        if img.get(attr):
                            url = img[attr]
                            # Handle srcset (take the largest)
                            if attr == 'data-srcset':
                                srcset = url.split(',')
                                if srcset:
                                    url = srcset[-1].split()[0]  # Last one is usually largest
                            
                            if url and self.is_valid_url(url) and url not in original_urls:
                                image_urls.append(url)
                                original_urls.add(url)
                                break
                                
            except Exception as e:
                self.logger.error(f"Error parsing with BeautifulSoup: {e}")
                
        # Clean up and filter URLs
        filtered_urls = []
        for url in image_urls:
            clean_url = self.clean_url(url)
            if clean_url and self.is_valid_image_url(clean_url) and clean_url not in filtered_urls:
                filtered_urls.append(clean_url)
                if len(filtered_urls) >= limit * 2:
                    break
                    
        self.logger.debug(f"Extracted {len(filtered_urls)} unique image URLs")
        return filtered_urls
    
    def clean_url(self, url):
        """
        Clean up URL by removing unnecessary parameters and fixing encoding
        """
        # Fix URL encoding
        url = urllib.parse.unquote(url)
        
        # Remove Google redirects
        if 'google.com' in url and '/url?q=' in url:
            url = re.search(r'/url\?q=([^&]+)', url)
            if url:
                url = url.group(1)

        # Fix truncated URLs
        if url.startswith("//"):
            url = "https:" + url
                
        return url
    
    def is_valid_url(self, url):
        """
        Check if a URL is valid
        """
        return bool(self.url_pattern.match(url))
    
    def is_valid_image_url(self, url):
        """
        Check if URL points to a valid image type
        """
        lower_url = url.lower()
        return any(ext in lower_url for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg'])
    
    def get_extension_from_url(self, url, default_extension='jpg'):
        """
        Extract file extension from URL or use default
        """
        url_lower = url.lower()
        if url_lower.endswith('.jpg') or url_lower.endswith('.jpeg'):
            return 'jpg'
        elif url_lower.endswith('.png'):
            return 'png'
        elif url_lower.endswith('.gif'):
            return 'gif'
        elif url_lower.endswith('.webp'):
            return 'webp'
        elif url_lower.endswith('.bmp'):
            return 'bmp'
        elif url_lower.endswith('.svg'):
            return 'svg'
        else:
            # Try to extract extension from url
            pattern = r'\.([a-zA-Z0-9]{3,5})(?:\?|&|$)'
            match = re.search(pattern, url_lower)
            if match and match.group(1) in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg']:
                ext = match.group(1)
                return 'jpg' if ext == 'jpeg' else ext
            return default_extension
    
    def get_filename(self, keyword, count, arguments, extension):
        """
        Generate filename for the image
        """
        keyword = keyword.replace(" ", "_")
        
        if arguments.get('no_numbering', False):
            filename = f"{keyword}.{extension}"
        else:
            filename = f"{keyword}_{count}.{extension}"
            
        if arguments.get('prefix'):
            filename = f"{arguments['prefix']}{filename}"
            
        return filename
    
    def save_image(self, url, path, arguments):
        """
        Download and save the image
        """
        if arguments.get('no_download', False):
            return True
            
        try:
            # Rotate user agents
            self.headers['User-Agent'] = random.choice(self.user_agents)
            self.headers['Referer'] = 'https://www.google.com/'
            
            # Create request
            request = urllib.request.Request(url, headers=self.headers)
            
            # Open image URL
            with urllib.request.urlopen(request, context=self.ssl_context, timeout=30) as response:
                # Check if response is valid
                if response.getcode() != 200:
                    self.logger.warning(f"Failed to download image (status code: {response.getcode()})")
                    return False
                
                # Get image data
                image_data = response.read()
                
                # Check if it's a valid image file
                if len(image_data) < 100:  # Too small to be a valid image
                    self.logger.warning(f"Image too small ({len(image_data)} bytes), skipping")
                    return False
                
                # Get file size
                file_size = len(image_data)
                
                # Print file size if requested
                if arguments.get('print_size', False):
                    self.logger.info(f"Image size: {self.format_size(file_size)}")
                
                # Save the image
                with open(path, 'wb') as f:
                    f.write(image_data)
                    
                return True
                
        except urllib.error.HTTPError as e:
            self.logger.warning(f"HTTP Error downloading {url}: {e.code} {e.reason}")
            return False
        except urllib.error.URLError as e:
            self.logger.warning(f"URL Error downloading {url}: {e.reason}")
            return False
        except Exception as e:
            self.logger.warning(f"Error downloading {url}: {e}")
            return False
    
    def format_size(self, size_bytes):
        """
        Format bytes to human-readable size
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

# Main implementation for the command-line tool
def main():
    parser = argparse.ArgumentParser(description='Download images from Google Images')
    parser.add_argument('-k', '--keywords', help='Keywords to search for', default="a photo of a cat")
    parser.add_argument('-l', '--limit', help='Maximum number of images to download', type=int, default=10)
    parser.add_argument('-o', '--output_directory', help='Main directory to save images', default='downloads')
    parser.add_argument('-i', '--image_directory', help='Sub-directory for the images')
    parser.add_argument('-f', '--format', help='Image format to save as', default='jpg')
    parser.add_argument('-s', '--size', help='Image size (large, medium, icon)')
    parser.add_argument('-c', '--color', help='Filter by color (red, orange, yellow, green, blue, purple, pink, white, gray, black, brown)')
    parser.add_argument('-t', '--type', help='Image type (face, photo, clipart, lineart, animated)')
    parser.add_argument('-d', '--delay', help='Delay between downloads in seconds', type=int)
    parser.add_argument('--print_urls', help='Print image URLs', action='store_true')
    parser.add_argument('--print_size', help='Print image sizes', action='store_true')
    parser.add_argument('--no_download', help='Skip actual downloading', action='store_true')
    parser.add_argument('--no_numbering', help='Do not add numbers to filenames', action='store_true')
    parser.add_argument('--prefix', help='Add prefix to filenames')
    parser.add_argument('--silent_mode', help='Suppress output', action='store_true')
    parser.add_argument('--debug', help='Enable debug logging', action='store_true')
    
    args = parser.parse_args()
    
    # Convert args to dict
    arguments = vars(args)
    
    # Download images
    downloader = GoogleImageCrawler(debug=args.debug)
    image_paths = downloader.download(arguments)
    
    # Print results
    if image_paths:
        print(f"\nSuccessfully downloaded {len(image_paths)} images:")
        for path in image_paths:
            print(f"  - {path}")
    else:
        print("\nNo images were downloaded.")

# Example usage as a module
class googleimagesdownload:
    def __init__(self, debug=False):
        self.downloader = GoogleImageCrawler(debug=debug)
        
    def download(self, arguments):
        """
        Download images based on the provided arguments
        """
        return self.downloader.download(arguments)

# Entry point
if __name__ == "__main__":
    main()