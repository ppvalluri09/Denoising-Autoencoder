import torch
from models import *
from prep_data import *
import torch.optim as optim
import pickle
import numpy as np

train_loader = get_data()
# print(next(iter(train_loader))[0].shape)

# autoencoder = AutoEncoder(784, 5)
autoencoder = AutoEncoderConv(1)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

def train(net, criterion, optimizer, train, EPOCHS):
    print('Training...')
    loss_history = []
    timeseries_img = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        net.train()
        for i, (x, _) in enumerate(train):
            optimizer.zero_grad()

            inp = x + (0.5*torch.randn(*x.shape))
            out = net(inp.float())
            loss = criterion(out, x.float())
            loss.backward()

            optimizer.step()

            if i % 100 == 99:
                # print(f'Epoch {epoch+1}, Batch [{i+1}/{len(train)}], Loss {loss.item()}')
                net.eval()
                with torch.no_grad():
                    val_out = net(inp.float())
                    val_loss = criterion(val_out, x.float())
                    loss_history.append([loss.item(), val_loss.item()])
                    timeseries_img.append(val_out)

                print(f'Epoch {epoch+1}, Batch [{i+1}/{len(train)}], Training Loss {loss.item()}, Validation Loss {val_loss.item()}')

    return net, loss_history, timeseries_img

TRAIN = True
if TRAIN:
    autoencoder, loss_history, timeseries_img = train(autoencoder, criterion, optimizer, train_loader, 10)
    torch.save(autoencoder.state_dict(), './models/denoiser_conv.pt')
    print('Model Saved')
    with open("./eval/generated_conv.pickle", "wb") as f:
        pickle.dump(timeseries_img, f)
    with open("./eval/loss_history_conv.pickle", "wb") as f:
        pickle.dump(np.array(loss_history), f)
    print('Saved Intermediate Results')
else:
    autoencoder = autoencoder.load_state_dict(torch.load('./models/denoiser_conv.pt'))
    print('Model Loaded')
