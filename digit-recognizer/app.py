from Classifier import Classifier
from PIL import Image
from torch import nn, save, load, argmax, tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor

trainingSet = datasets.MNIST('./', train=True, transform=ToTensor(), download=True)
trainingData = DataLoader(dataset=trainingSet, batch_size=32)

# WIP
model_instance = Classifier().to('cpu')
optimizer = Adam(model_instance.parameters(), lr=.001)
loss_function = nn.CrossEntropyLoss()

# for epoch in range(10):
#     for data, label in trainingData:
#         data, label = data.to('cpu'), label.to('cpu')
#         res = model_instance(data)
#         loss = loss_function(res, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch} loss is {loss.item()}')

# with open('model_state.pt', 'wb') as f:
#     save(model_instance.state_dict(), f)

with open('model_state.pt', 'rb') as f:
    model_instance.load_state_dict(load(f))

    img = Image.open('img_3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    print(img_tensor[0][0])
    print(argmax(model_instance(img_tensor)))