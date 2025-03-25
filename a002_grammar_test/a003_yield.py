def y_func():
    for i in range(10):
        yield i


print(type(y_func))
print(type(y_func()))

for item in y_func():
    print(item)
