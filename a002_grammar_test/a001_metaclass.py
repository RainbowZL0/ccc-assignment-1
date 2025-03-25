def __init__method(self, value):
    self.value = value  # 这是一个实例属性


def show(self):
    print("Instance value is:", self.value)


# 使用 type() 创建一个类 MyClass
MyClass = type(
    'MyClass',
    (),
    {
        '__init__method': __init__method,
        'lambda_method': lambda x: x*2,
        'show': show,
    }
)

# 创建 MyClass 的实例并调用方法
obj = MyClass(42)
obj.show()  # 输出: Instance value is: 42
