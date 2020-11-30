# Send Requests to Triton Inference Server with FastNN Client 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aychang95/fastnn/blob/main/notebooks/inference_client.ipynb)


# Examples:

### **"distilbert-squad" Model**


```python
from fastnn.processors.nlp.question_answering import TransformersQAProcessor

context = ["""Albert Einstein was born at Ulm, in Württemberg, Germany, on March 14, 1879. Six weeks later the family moved to Munich, where he later on began his schooling at the Luitpold Gymnasium. 
Later, they moved to Italy and Albert continued his education at Aarau, Switzerland and in 1896 he entered the Swiss Federal Polytechnic School in Zurich to be trained as a teacher in physics and mathematics. 
In 1901, the year he gained his diploma, he acquired Swiss citizenship and, as he was unable to find a teaching post, he accepted a position as technical assistant in the Swiss Patent Office. In 1905 he obtained his doctor’s degree.
During his stay at the Patent Office, and in his spare time, he produced much of his remarkable work and in 1908 he was appointed Privatdozent in Berne. In 1909 he became Professor Extraordinary at Zurich, in 1911 Professor of 
Theoretical Physics at Prague, returning to Zurich in the following year to fill a similar post. In 1914 he was appointed Director of the Kaiser Wilhelm Physical Institute and Professor in the University of Berlin. He became a 
German citizen in 1914 and remained in Berlin until 1933 when he renounced his citizenship for political reasons and emigrated to America to take the position of Professor of Theoretical Physics at Princeton. He became a United 
States citizen in 1940 and retired from his post in 1945. After World War II, Einstein was a leading figure in the World Government Movement, he was offered the Presidency of the State of Israel, which he declined, and he 
collaborated with Dr. Chaim Weizmann in establishing the Hebrew University of Jerusalem. Einstein always appeared to have a clear view of the problems of physics and the determination to solve them. He had a strategy of 
his own and was able to visualize the main stages on the way to his goal. He regarded his major achievements as mere stepping-stones for the next advance. At the start of his scientific work, Einstein realized the 
inadequacies of Newtonian mechanics and his special theory of relativity stemmed from an attempt to reconcile the laws of mechanics with the laws of the electromagnetic field. He dealt with classical problems of 
statistical mechanics and problems in which they were merged with quantum theory: this led to an explanation of the Brownian movement of molecules. He investigated the thermal properties of light with a low radiation
density and his observations laid the foundation of the photon theory of light. In his early days in Berlin, Einstein postulated that the correct interpretation of the special theory of relativity must also furnish a
theory of gravitation and in 1916 he published his paper on the general theory of relativity. During this time he also contributed to the problems of the theory of radiation and statistical mechanics."""]

query = ["When was Einstein born?"]

# Specify tokenizer for encoding

model_name_or_path = "distilbert-base-cased-distilled-squad"

processor = TransformersQAProcessor(model_name_or_path=model_name_or_path)

examples, features, dataloader = processor.process_batch(query=query*8, context=context*8, mini_batch_size=8, use_gpu=True)

```


```python
import torch
import numpy as np

from fastnn.client import FastNNClient

client = FastNNClient(url="127.0.0.1:8000", model_name="distilbert-squad", model_version="1")

#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        response = client.request(batch) 
        start_logits = response.as_numpy('output__0')
        start_logits = np.asarray(start_logits, dtype=np.float32)
        end_logits = response.as_numpy('output__1')
        end_logits = np.asarray(end_logits, dtype=np.float32)
        example_indices = response.as_numpy('output__2')
        example_indices = np.asarray(example_indices, dtype=np.int64)
        
        output = (torch.from_numpy(start_logits), torch.from_numpy(end_logits), torch.from_numpy(example_indices))
        all_outputs.append(output)
all_outputs

```


```python
from fastnn.processors.cv.object_detection import ObjectDetectionProcessor

# COCO dataset category names
label_strings = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

processor = ObjectDetectionProcessor(label_strings=label_strings)

# Replace "img_dir_path" with root directory of .png or .jpeg images
dataloader = processor.process_batch(dir_path="./img_dir_path", mini_batch_size=2, use_gpu=False)


```

### **"fasterrcnn-resnet50" Model**


```python
import torch
import numpy as np

from fastnn.client import FastNNClient

client = FastNNClient(url="127.0.0.1:8000", model_name="fasterrcnn-resnet50-cpu", model_version="1")

#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        response = client.request(*batch) 
        boxes = response.as_numpy('output__0')
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = response.as_numpy('output__1')
        labels = np.asarray(labels, dtype=np.int64)
        scores = response.as_numpy('output__2')
        scores = np.asarray(scores, dtype=np.float32)
        
        output = (torch.from_numpy(boxes), torch.from_numpy(labels), torch.from_numpy(scores))
        all_outputs.append(output)
        
all_outputs
```
