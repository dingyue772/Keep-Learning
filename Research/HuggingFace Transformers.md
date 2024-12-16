# Basic Skills
## 使用Pipeline进行推理

代码
```python
# pipeline创建
from transformers import pipeline
transcriber = pipeline(task="automatic-speech-recognition") # 指定推理任务
# 传入输入，执行推理任务
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

> tips: pipeline很适合用来做测试，写好的脚本不用修改代码

通用参数
`device`
`device=n`用于指定模型所在设备，如果安装`accelerate`包后使用`device_map="auto"`自动确定模型权重的加载和存储，注意`device`和`device_map`不要同时指定
`batch_size`
批量推理
```python
transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
```
模型推理时的参数
安装 `bitsandbytes` 并添加参数 `load_in_8bit=True`，通过`model_kwargs`传递参数
```python
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```
任务特定参数
```python
# 这里使用transformers.AutomaticSpeechRecognitionPipeline.**call**()方方法中的 `return_timestamps` 参数
transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

在数据集上使用pipelines
通过Huggingface的Datasets以迭代器的形式加载数据，在GPU上处理数据的同时开始获取数据
```python
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```

## 使用AutoClass加载模型
加载NLP任务中的`tokenizer`
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

加载cv任务中的图像预处理器 `image processor`
```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

加载音频任务中的`feature extractor`
```python
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
```

加载多模态任务中的`processor`
```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

加载给定任务的预训练模型`AutoModelFor`
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

## 预处理数据
利用前面加载的`tokenizer`、`feature extractor`、`ImageProcessor`、`Processor`将文本、语音、图像和多模态输入转换为成批量的张量以供模型使用

文本数据预处理
```python
# tokenizer加载
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
# tokenize文本数据
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input) 
# 解码input_ids
tokenizer.decode(encoded_input["input_ids"])
```
`tokenizer`返回一个包含三个重要对象的字典：
- `input_ids`: 与句子中每个`token`对应的索引
- `attention_masks`: 指示模型是否关注一个`token`
- `token_type_ids`: 指示当前`token`为输入序列中的哪一个，因为tokenize对象可为一个list包含多个序列

填充
使用参数`padding=True`
当输入序列列表中句子长度不同时，tokenizer会在较短的句子中添加一个特殊的`padding token`，以确保张量是矩形

截断
使用参数`truncation=True`
将序列阶段到模型能接受的最大长度

张量构建
使用参数`return_tensors="pt"`返回PyTorch框架下的张量类型

示例代码
```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)
```

音频处理

音频数据集
```python
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

dataset[0]["audio"]
```
数据集加载，访问 `audio` 列的第一个元素以查看输入。调用 `audio` 列会自动加载和重新采样音频文件，这里将返回以下三个对象
- `array` 是加载的语音信号 - 并在必要时重新采为`1D array`
- `path` 指向音频文件的位置
- `sampling_rate` 是每秒测量的语音信号数据点数量
根据模型训练数据，对数据集进行采样率提升的重采样
```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

`feature extractor`加载并处理语音信号
```python
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# 音频采样后的语音信号预处理
audio_input = [dataset[0]["audio"]["array"]]
feature_extractor(audio_input, sampling_rate=16000)

# 使用截断和填充处理不同长度的音频信号
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
```

图片处理

图片数据集加载
```python
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:100]")
# 图像查看
dataset[0]["image"]
```

`image processor`加载
```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

使用`torchvision.transforms`对图像进行增强处理，这里使用`RandomResizedCrop`和`ColorJitter`处理图像，并使用`Compose`连接图像处理操作
```python
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

使用`ImageProcessor`进行图像标准化处理
```python 
def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples

dataset.set_transform(transforms)
```

填充处理
```python 
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```

多模态处理，这里使用的是自动语音识别任务（文本、语音多模态）

数据集加载
```python 
from datasets import load_dataset

lj_speech = load_dataset("lj_speech", split="train")

# 数据集查看
lj_speech[0]["audio"]

lj_speech[0]["text"]

# 音频重采样
lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

`Processor`加载
```python 
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

`Processor`处理，将包含在 `array` 中的音频数据处理为 `input_values`，并将 `text` 标记为 `labels`
```python 
def prepare_dataset(example):
    audio = example["audio"]

    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

    return example

prepare_dataset(lj_speech[0])
```

## 微调预训练模型
数据集加载
```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
```

`tokenizer`处理文本
```python 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

模型加载
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

使用`PyTorch Trainer`训练模型
```python
# 评估
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 训练超参数设定，使用TrainingArguments
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# 创建训练器，训练器中包含模型、训练参数、训练集、测试集、评估函数
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()
```

