from Medical_Dataset import *


def folder_name_gen(prefix, kind, dye, mag):
    subtype_folder = prefix + kind + "/" + dye
    subtype_folder_content = os.listdir(subtype_folder)

    dir_list = []
    for i in range(0, len(subtype_folder_content)):
        current_index_folder = subtype_folder + "/" + subtype_folder_content[i]
        current_index_folder_content = os.listdir(current_index_folder)

        for j in range(0, len(current_index_folder_content)):
            current_mag_folder = current_index_folder + "/" + current_index_folder_content[j]
            image_folder = current_mag_folder + "/" + mag + "/"
            dir_list.append(image_folder)

    return dir_list


magnification = "400X"
dye = "SOB"
main_path = "../Datasets/BreaKHis_v1/histology_slides/breast/"
img_type = "png"

area_size = 64
batch_size = 32
training_rate = 0.7

read_from_saved = bool(int(sys.argv[1]))
resize_rate = float(sys.argv[2])
load_list = bool(int(sys.argv[3]))
initial_learning_rate = float(sys.argv[4])
if_aug = bool(int(sys.argv[5]))
if_ten_crop = bool(int(sys.argv[6]))

if read_from_saved:
    benign_x = np.load("benign_box_" + magnification + ".npy")
    malignant_x = np.load("malignant_box_" + magnification + ".npy")
else:
    benign_folder_list = folder_name_gen(main_path, "benign", dye, magnification)
    malignant_folder_list = folder_name_gen(main_path, "malignant", dye, magnification)

    benign_x = process_and_concat(read_img_dirlist(benign_folder_list, img_type),
                                  process_op=size_filter)
    malignant_x = process_and_concat(read_img_dirlist(malignant_folder_list,
                                     img_type), process_op=size_filter)

    # save for future use
    np.save("benign_box_" + magnification + ".npy", benign_x)
    np.save("malignant_box_" + magnification + ".npy", malignant_x)

# resize
benign_x = resize_img_mat(benign_x, resize_rate, whether_readable=True)
malignant_x = resize_img_mat(malignant_x, resize_rate, whether_readable=True)

length = benign_x.shape[1]
width = benign_x.shape[2]
channels = benign_x.shape[3]

print(benign_x.shape)
print(malignant_x.shape)

total_p, train_p, total_n, train_n = list_size_calc(malignant_x, benign_x, training_rate)

print(train_n)
print(train_p)

train_list_p, test_list_p, train_list_n, test_list_n = get_balanced_load_list(load_list, train_p, total_p, train_n, total_n)

train_image_p = malignant_x[train_list_p]
train_image_n = benign_x[train_list_n]
test_image_p = malignant_x[test_list_p]
test_image_n = benign_x[test_list_n]

print(train_image_p.shape)
print(train_image_n.shape)
print(test_image_p.shape)
print(test_image_n.shape)

train_label_p = np.ones(len(train_image_p), dtype=int)
train_label_n = np.zeros(len(train_image_n), dtype=int)
test_label_p = np.ones(len(test_image_p), dtype=int)
test_label_n = np.zeros(len(test_image_n), dtype=int)

train_image = common_concat([train_image_p, train_image_n])
test_image = common_concat([test_image_p, test_image_n])

train_label = common_concat([train_label_p, train_label_n])
test_label = common_concat([test_label_p, test_label_n])

resize_dim = (int(length * resize_rate), int(width * resize_rate))

# set up transformations according to the setting
if if_ten_crop:
    tran_test_set = [#transforms.Resize(resize_dim, interpolation=Image.ANTIALIAS),
                     transforms.TenCrop(area_size)]
    if if_aug:
        tran_test_set.append(transforms.Lambda(lambda crops:
                             torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))
                             (transforms.ToTensor()(crop)) for crop in crops])))
    else:
        tran_test_set.append(transforms.Lambda(lambda crops:
                             torch.stack([transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))
                             (transforms.ToTensor()(crop)) for crop in crops])))
else:
    tran_test_set = [#transforms.Resize(resize_dim, interpolation=Image.ANTIALIAS),
                     transforms.CenterCrop(area_size)]
    if if_aug:
        tran_test_set.extend([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                   (0.2023, 0.1994, 0.2010))])
    else:
        tran_test_set.extend([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5))])

if if_aug:
    tran_train_set = [#transforms.Resize(resize_dim, interpolation=Image.ANTIALIAS),
                      transforms.RandomCrop(area_size, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.2023, 0.1994, 0.2010))]
else:
    tran_train_set = [#transforms.Resize(resize_dim, interpolation=Image.ANTIALIAS),
                      transforms.CenterCrop(area_size),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

transform_train = transforms.Compose(tran_train_set)
transform_test = transforms.Compose(tran_test_set)

trainset = MedicalImages(data=train_image, labels=train_label, transform=transform_train)
testset = MedicalImages(data=test_image, labels=test_label, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

#net = InceptPreActResNet18(num_classes=2)
net = PreActResNet18(num_classes=2)
#net = WideResNet16_8(num_classes=2)
net.cuda()
print(net)

training(net, 200, initial_learning_rate,
         trainloader, testloader, if_ten_crop=if_ten_crop,
         test_device='gpu', loss_interval=6, accum_grad_num=4)
