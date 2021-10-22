# PYTORCH随机权重比特翻转攻击
> Implement random weight's bit flip attack based on PytorchFI

[PytorchFI](https://github.com/pytorchfi/pytorchfi)是MIT的一个研究小组推出的故障注入框架

但该框架的文档更新不是很详细，参数名也不是很友好

因此，我在此基础上再包了一层

用法：

```python
Attack(model=vgg_16,batch=80,path=path,bitlen=32,perlayer_error_num=3000)
```

[具体文件](https://github.com/Force1ess/Random_WeightAttact-PytorchFi/blob/main/main.py)
