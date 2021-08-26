label = torch.tensor(labels)
label = label.unsqueeze(0)
label = label/torch.sum(label)
weights = 1.0 / label
loss_fn = torch.nn.CrossEntropyLoss(weight=weights,reduction='sum').to(device)

"""or
label = torch.tensor(labels)
label = label.unsqueeze(0)
label = torch.sum(label) / label
label"""