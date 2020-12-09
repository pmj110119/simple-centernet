# simple-centernet

> 项目需要用centernet作为pipeline中一部分，但centernet作者发布版本功能太多，框架过于复杂，用起来挺爽的但对于二次开发太臃肿了。索性花半天写了个纯净版本，支持目标检测和关键点检测。

### 1.install

simple-centernet: https://github.com/pmj110119/simple-centernet.git

```
git clone https://github.com/pmj110119/simple-centernet.git
```

install packages

```bash
pip install -r requirements.txt
```

### 2. train

prepare COCO dataset and then start tringing

```python
python train.py 
```

