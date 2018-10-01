import json
import csv
import operator

# Load original data
with open('data.json') as f:
    data = json.load(f)

# Total number of threads (Q&A) - 30702
print(len(data['threads']))

# Show the structure of the dataset
print(json.dumps(data['threads'][0], indent=2, sort_keys=False))

# Inspect the types of data we are dealing with
print(type(data))
print(type(data['threads']))
print(type(data['threads'][0]))

# Show how we pick out desired fields
print(data['threads'][0]['title'])
print(data['threads'][0]['question']['text'])
print(data['threads'][0]['tags'])

# Number of posts with one or more tag(s)
num_tag_post = 0

for item in data['threads']:
    if item['tags']:
        num_tag_post += 1

print(num_tag_post)

# Made three different counters that identified unique tags and counted number of occurrences. This is one of them.
unique_tag = {}

for item in data['threads']:
    if type(item['tags']) == list:
        for tags in item['tags']:
            tags = tags.split()
            for tag in tags:
                if tag not in unique_tag.keys():
                    unique_tag[tag] = 1
                else:
                    unique_tag[tag] += 1

print(sorted(unique_tag.items(), key=operator.itemgetter(1), reverse=True))
print(len(unique_tag))

# Upon inspection, I discovered that it was possible to construct a dataset that deals with different
# configuration possibilities of the pins on the microcontrollers. Features = Title + text. Labels = tag.
# The size of the data set was limited by the tag with fewest occurrences.
valid_tags = ['uart', 'spi', 'adc', 'twi', 'timer', 'pwm', 'i2c', 'interrupt']
valid_tag_count = {'uart': 0, 'spi': 0, 'adc': 0, 'twi': 0, 'timer': 0, 'pwm': 0, 'i2c': 0, 'interrupt': 0}
newData = []

for item in data['threads']:
    if type(item['tags']) == list:
        for tags in item['tags']:
            tags = tags.split()
            for tag in tags:
                if tag in valid_tags:
                    if valid_tag_count[tag] < 150:
                        valid_tag_count[tag] += 1
                        title = item['title']
                        question = item['question']['text']
                        text = "{0} {1}".format(title, question)
                        post = {'text': text, 'tag': tag}
                        newData.append(post)

print(len(newData))

print(json.dumps(newData[:3], indent=2, sort_keys=False))

with open('newData.json', 'w') as f:
    json.dump(newData, f, indent=2)


