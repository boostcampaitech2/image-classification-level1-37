train_transform = T.Compose([
    T.ToPILImage(),
    T.CenterCrop([300,250]),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
])

valid_transform = T.Compose([
    T.ToPILImage(),
    T.CenterCrop([300,250]),
    T.ToTensor(),
    T.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
])
