# Model Zoo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aychang95/fastnn/blob/master/notebooks/model_zoo.ipynb)

## Available Serialized Models

### Natural Language Processing:

| Model Name | Size | Origin |
| ---------- | ---- | ------ |
| distilbert-squad | 249 MB | [Transformers](https://huggingface.co/distilbert-base-cased-distilled-squad) |
| distilbert-squad-cpu | 249 MB| [Transformers](https://huggingface.co/distilbert-base-cased-distilled-squad) |
| bert-large-cased-whole-word-masking-finetuned-squad | 1.24 GB | [Transformers](https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad) |
| bert-large-cased-whole-word-masking-finetuned-squad-cpu | 1.24 GB | [Transformers](https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad) |
| deepset.roberta-base-squad2 | 474 MB | [Transformers](https://huggingface.co/deepset/roberta-base-squad2) |
| deepset.roberta-base-squad2-cpu | 474 MB | [Transformers](https://huggingface.co/deepset/roberta-base-squad2) |
| deepset.bert-large-uncased-whole-word-masking-squad2 | 1.25 GB | [Transformers](https://huggingface.co/deepset/bert-large-uncased-whole-word-masking-squad2) |
| deepset.bert-large-uncased-whole-word-masking-squad2-cpu | 1.25 GB | [Transformers](https://huggingface.co/deepset/bert-large-uncased-whole-word-masking-squad2) |
| mrm8488.bert-base-portuguese-cased-finetuned-squad-v1-pt | 414 MB | [Transformers](https://huggingface.co/mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt) |
| mrm8488.bert-base-portuguese-cased-finetuned-squad-v1-pt-cpu | 414 MB | [Transformers](https://huggingface.co/mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt) |


### Computer Vision
| Model Name | Size | Origin |
| ---------- | ---- | ------ |
| fasterrcnn-resnet50-cpu | 160 MB | [TorchVision]() |



## Run Inference Programatically on Models using `Processor` and FastNN Modules

### Examples:

#### **SQuAD Question Answering**

Run pre-requisite data processing steps. This will create compatible input data required for tracing our models in the next step.

See `TransformersQAProcessor` section in Data Processing documentation page for more information.


```python
from fastnn.processors.nlp.question_answering import TransformersQAProcessor

context = ["""Albert Einstein was born at Ulm, in Württemberg, Germany, on March 14, 1879. Six weeks later the family moved to Munich, where he later on began his schooling at the Luitpold Gymnasium. 
Later, they moved to Italy and Albert continued his education at Aarau, Switzerland and in 1896 he entered the Swiss Federal Polytechnic School in Zurich to be trained as a teacher in physics and mathematics. 
In 1901, the year he gained his diploma, he acquired Swiss citizenship and, as he was unable to find a teaching post, he accepted a position as technical assistant in the Swiss Patent Office. In 1905 he obtained his doctor’s degree.
During his stay at the Patent Office, and in his spare time, he produced much of his remarkable work and in 1908 he was appointed Privatdozent in Berne. In 1909 he became Professor Extraordinary at Zurich, in 1911 Professor of 
Theoretical Physics at Prague, returning to Zurich in the following year to fill a similar post. In 1914 he was appointed Director of the Kaiser Wilhelm Physical Institute and Professor in the University of Berlin. He became a 
German citizen in 1914 and remained in Berlin until 1933 when he renounced his citizenship for political reasons and emigrated to America to take the position of Professor of Theoretical Physics at Princeton*. He became a United 
States citizen in 1940 and retired from his post in 1945. After World War II, Einstein was a leading figure in the World Government Movement, he was offered the Presidency of the State of Israel, which he declined, and he 
collaborated with Dr. Chaim Weizmann in establishing the Hebrew University of Jerusalem. Einstein always appeared to have a clear view of the problems of physics and the determination to solve them. He had a strategy of 
his own and was able to visualize the main stages on the way to his goal. He regarded his major achievements as mere stepping-stones for the next advance. At the start of his scientific work, Einstein realized the 
inadequacies of Newtonian mechanics and his special theory of relativity stemmed from an attempt to reconcile the laws of mechanics with the laws of the electromagnetic field. He dealt with classical problems of 
statistical mechanics and problems in which they were merged with quantum theory: this led to an explanation of the Brownian movement of molecules. He investigated the thermal properties of light with a low radiation
density and his observations laid the foundation of the photon theory of light. In his early days in Berlin, Einstein postulated that the correct interpretation of the special theory of relativity must also furnish a
theory of gravitation and in 1916 he published his paper on the general theory of relativity. During this time he also contributed to the problems of the theory of radiation and statistical mechanics."""]

query = ["When was Einstein born?"]

model_name_or_path = "distilbert-base-cased-distilled-squad"

processor = TransformersQAProcessor(model_name_or_path=model_name_or_path)

examples, features, dataloader = processor.process_batch(query=query*8, context=context*8, mini_batch_size=8, use_gpu=True)
```

Using processed input data, export your torch model to a model compatible with C++ programs.

See `BertQAModule` section in Exporting Models documentation page for more information.

We'll show an example of running inference on a model from the "model_repository".


```python
from fastnn.nn.question_answering import BertQAModule

pytorch_model = BertQAModule(model_name_or_path=model_name_or_path).to("cuda")
pytorch_model.eval()
```

Run inference on loaded torchscript model and pytorch model.


```python
import torch

torchscript_model = torch.jit.load("../../model_repository/distilbert-squad/1/model.pt")

#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        output = torchscript_model(*batch) 
        all_outputs.append(output)

#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        output = pytorch_model(*batch) 
        all_outputs.append(output)
```

Use Processor class to produce human legible data from model outputs.


```python
answers, n_best = processor.process_output_batch(
        outputs = all_outputs,
        examples = examples,
        features = features,
        n_best_size= 5,
        max_answer_length= 16,
        do_lower_case = False,
        verbose_logging= False,
        version_2_with_negative= False,
        null_score_diff_threshold = 0.0
)

answers, n_best
```

#### **Object Detection**

### **`FasterRCNNModule`**


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


```python
from fastnn.nn.object_detection import FasterRCNNModule

pytorch_model = FasterRCNNModule().to("cpu")
pytorch_model.eval()
```

Run inference on the loaded torchscript model as well as the pytorch model. Dataloader batch size for the torchscript model must be 1.


```python
import torch

loaded_torchscript_model = torch.jit.load("../../model_repository/fasterrcnn-resnet50-cpu/1/model.pt")

#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        output = pytorch_model(*batch) 
        all_outputs.append(output)
        
#%%timeit
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        output = loaded_torchscript_model(batch[0][0]) 
        all_outputs.append(output)
```


```python
# Replace "img_dir_path" with root directory of .png or .jpeg images
dataset = processor.process(dir_path="./img_dir_path")
results = processor.process_output_batch(outputs=all_outputs, dataset=dataset)
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
plt.imshow(results[1][0][1])
```

