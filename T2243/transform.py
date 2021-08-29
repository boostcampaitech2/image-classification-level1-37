train_transform = T.Compose([
    T.ToPILImage(),
    T.CenterCrop([300,250]),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=(0.56, 0.51, 0.48), std=(0.22, 0.24, 0.25))
])

valid_transform = T.Compose([
    T.ToPILImage(),
    T.CenterCrop([300,250]),
    T.ToTensor(),
    T.Normalize(mean=(0.56, 0.51, 0.48), std=(0.22, 0.24, 0.25))
])

