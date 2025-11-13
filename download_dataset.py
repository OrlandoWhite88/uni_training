import gdown

# Google Drive file ID
file_id = "1iTH0uABWsppQmY348YeqZWJ_-X5YXgWw"

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output = "dataset.jsonl"

# Download the file
print(f"Downloading dataset to {output}...")
gdown.download(url, output, quiet=False)
print(f"Download complete! File saved as {output}")
