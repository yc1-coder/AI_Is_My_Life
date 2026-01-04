'''
for epoch in range(num_epochs):
    #train_loader 是生成器，每次可以提取一个小批次数据
    for batch_idx,(input,labels) in enumberate(train_loader):
        #1.正向传播
        outputs = model(inputs)
        #2.计算损失
        loss = criterion(outputs,labels)
        #3.清空梯度，把上一轮的梯度清空
        optimizer.zero_grad()
        #4.反向传播
        loss.backward()
        #5.更新参数
        optimizer.step()
#清空梯度在反向传播之前即可
'''