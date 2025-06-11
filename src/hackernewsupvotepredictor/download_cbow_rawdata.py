import urllib.request

url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
response = urllib.request.urlopen(url)
data = response.read()      # a bytes object
text = data.decode('utf-8')

with open('data/text8', 'w') as file:
    file.write(text)
