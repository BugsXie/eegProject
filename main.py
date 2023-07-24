from config import *

train_data = CostumDataset("./data/extract_data/train")
val_data = CostumDataset("./data/extract_data/val")

train_dataloader = DataLoader(train_data)
val_dataloader = DataLoader(val_data)

for epoch in range(num_epochs):
    model.train()

    for data, label in train_dataloader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label.long())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for data, label in val_dataloader:
            data = data.to(device)
            label = label.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
        accuracy = total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}")
