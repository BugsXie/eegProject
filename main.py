from torch.utils.tensorboard import SummaryWriter

from config import *

train_data = CostumDataset("./data/extract_data/train")
val_data = CostumDataset("./data/extract_data/val")

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)

total_train_step = 0
total_val_step = 0
start_time = time.time()
writer = SummaryWriter(r".\loss_train_2_8_5")

for epoch in range(num_epochs):
    print("--------第{}轮训练开始--------".format(epoch + 1))
    model.train()
    total_correct_train = 0
    total_samples_train = 0

    for data, label in train_dataloader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label.long())
        loss.backward()

        optimizer.step()
        total_train_step += 1
        if total_train_step % 10 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            end_time = time.time()
            print("耗时：{}s".format(end_time - start_time))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        _, predicted = torch.max(output, dim=1)
        total_correct_train += (predicted == label).sum().item()
        total_samples_train += label.size(0)
    accuracy_train = total_correct_train / total_samples_train
    writer.add_scalar("train_auc", accuracy_train, total_train_step)

    print("训练集准确率", f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {accuracy_train:.4f}")

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

        print("测试集准确率", f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}")
        writer.add_scalar("total_test_accuracy", accuracy, total_val_step)
        total_val_step += 1
writer.close()