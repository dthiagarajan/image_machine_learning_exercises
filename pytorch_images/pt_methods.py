import time
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from Cifar_Net import *
from PreAct_Res_Net import *
from Wide_Res_Net import *
from Pyramid_Net import *

sys.path.append('../Common_Code/')
from python_methods_image import *

weight_decay = 5e-4
max_norm_clip = 20


# check nan in mat
def check_mat_nan(mat):
    return (torch.sum(torch.isnan(mat)) > 0)


def nan_to_zero_helper(mat, regenerator):
    if check_mat_nan(mat):
        replacement = regenerator(mat.size())
        # wrap replacement in nan_to_zero, prevent from regen
        # nans! use cpu as device allows
        replacement = nan_to_zero(mat=replacement,
                                  regenerator=regenerator)

        return replacement

    return mat


# modify nan mats
def nan_to_zero(mat, device="gpu", regenerator=torch.zeros):
    if check_mat_nan(mat):
        print("observe an nan in the gradient!!!!!")

        replacement = nan_to_zero_helper(mat=replacement,
                                         regenerator=regenerator)

        if device == "gpu":
            replacement = replacement.cuda()
        else:
            replacement = replacement.cpu()

        return replacement

    return mat


def imshow(img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_acc(model, loader, if_ten_crop, device, set_name):
    start = time.time()

    max_correct = 0
    mean_correct = 0
    min_correct = 0

    total = 0
    similarity_collection = []

    if set_name == "test":
        set_info = "test images"
        in_test = True
    else:
        set_info = "train images"
        in_test = False

    execute_tc = if_ten_crop and in_test

    # copy model to cpu
    if device == 'cpu':
        model.cpu()

    # turn on eval mode
    model.eval()

    for i, data in enumerate(loader, 0):
        images, labels = data

        if device == 'gpu':
            images, labels = images.cuda(), labels.cuda()

        if execute_tc:
            batch_size, ncrops, c, h, w = images.size()
            images = images.view(-1, c, h, w)

        outputs = model(images).data

        if ('memorize' in model.__class__.__dict__):
            keys = model._model(images)
            similarity = model.memory_bank.cosine_sim_weight(keys).clone().data
            similarity_collection.append(similarity.cpu())

        if execute_tc:
            max_outputs = outputs.view(batch_size, ncrops, -1).max(1)[0]
            mean_outputs = outputs.view(batch_size, ncrops, -1).mean(1)
            min_outputs = outputs.view(batch_size, ncrops, -1).min(1)[0]
        else:
            max_outputs = outputs
            mean_outputs = outputs
            min_outputs = outputs

        _, max_predicted = torch.max(max_outputs, 1)
        _, mean_predicted = torch.max(mean_outputs, 1)
        _, min_predicted = torch.max(min_outputs, 1)

        total += labels.size(0)

        max_correct += (max_predicted == labels).sum()
        mean_correct += (mean_predicted == labels).sum()
        min_correct += (min_predicted == labels).sum()

    if ('memorize' in model.__class__.__dict__):
        similarity_collection = torch.cat(similarity_collection, dim=0)

    # turn back to train model
    model.train()

    # copy model back
    if device == 'cpu':
        model.cuda()

    max_acc = (100 * max_correct.float() / total).item()
    mean_acc = (100 * mean_correct.float() / total).item()
    min_acc = (100 * min_correct.float() / total).item()

    print('Accuracy of the network on the ' + set_info + ' with max crop pool: ' + str(max_acc) + '%')
    print('Accuracy of the network on the ' + set_info + ' with mean crop pool: ' + str(mean_acc) + '%')
    print('Accuracy of the network on the ' + set_info + ' with min crop pool: ' + str(min_acc) + '%')

    end = time.time()
    print("this takes a total of: " + str(end - start) + " seconds \n")

    return [max_acc, mean_acc, min_acc], similarity_collection


def manual_adjusted_scheme_1(current_epoch):
    if current_epoch < 30:
        decay = 1
        grow = 0
        norm_clip_rate = 1
        write_rate_adjust = 1
    elif current_epoch < 45:
        decay = 3e-1
        grow = 1e-3
        norm_clip_rate = 1 #5e-2
        write_rate_adjust = 1
    elif current_epoch < 60:
        decay = 1e-1
        grow = 5e-3
        norm_clip_rate = 1 #1e-1
        write_rate_adjust = 1
    elif current_epoch < 75:
        decay = 5e-2
        grow = 2e-2
        norm_clip_rate = 1 #2e-1
        write_rate_adjust = 1
    elif current_epoch < 90:
        decay = 1e-2
        grow = 8e-2
        norm_clip_rate = 1 #5e-1
        write_rate_adjust = 1
    elif current_epoch < 120:
        decay = 3e-3
        grow = 3e-1
        norm_clip_rate = 1
        write_rate_adjust = 1
    elif current_epoch < 150:
        decay = 1e-3
        grow = 1
        norm_clip_rate = 1
        write_rate_adjust = 1
    else:
        decay = 3e-4
        grow = 1
        norm_clip_rate = 1
        write_rate_adjust = 1
    return decay, grow, norm_clip_rate, write_rate_adjust


def manual_adjusted_scheme_2(current_epoch):
    if current_epoch < 30:
        decay = 1
        grow = 0
        norm_clip_rate = 1
        write_rate_adjust = 1
    elif current_epoch < 60:
        decay = 2e-1
        grow = 1e-3
        norm_clip_rate = 1 #5e-2
        write_rate_adjust = 1
    elif current_epoch < 90:
        decay = 4e-2
        grow = 5e-3
        norm_clip_rate = 1 #1e-1
        write_rate_adjust = 1
    elif current_epoch < 120:
        decay = 8e-3
        grow = 2e-2
        norm_clip_rate = 1 #2e-1
        write_rate_adjust = 1
    else:
        decay = 2e-3
        grow = 1
        norm_clip_rate = 1
        write_rate_adjust = 1
    return decay, grow, norm_clip_rate, write_rate_adjust


def cifar_test_scheme(current_epoch):
    if current_epoch < 60:
        decay = 1
        grow = 0
        norm_clip_rate = 1
        write_rate_adjust = 1
    elif current_epoch < 120:
        decay = 2e-1
        grow = 1e-2
        norm_clip_rate = 1 #5e-2
        write_rate_adjust = 1
    elif current_epoch < 160:
        decay = 4e-2
        grow = 1e-1
        norm_clip_rate = 1 #1e-1
        write_rate_adjust = 1
    else:
        decay = 8e-3
        grow = 1
        norm_clip_rate = 1
        write_rate_adjust = 1
    return decay, grow, norm_clip_rate, write_rate_adjust


# hook for guarding nan in gradient
def gradient_nan_guard_hook(module, grad_in, grad_out):
    new_grad_in = []
    for i in range(0, len(grad_in)):
        new_grad_in.append(nan_to_zero(grad_in[i]))

    return tuple(new_grad_in)


# abbreviation: ilr -> initial learning rate
# fybl1lr -> final y bank L1 loss rate
# cybl1lr -> current y bank L1 loss rate
# bndt -> bank normalization during training
def train_action(model, current_epoch, ilr, criterion,
                 trainloader, testloader, if_ten_crop,
                 test_device, loss_interval, bndt,
                 accum_grad_num, backward_all,
                 norm_clip_option, fname,
                 final_write_rate, fybl1lr):
    start = time.time()
    print("Epoch %d: " % current_epoch)

    #decay, grow, norm_clip_rate, write_rate_adjust = manual_adjusted_scheme_1(current_epoch)
    decay, grow, norm_clip_rate, write_rate_adjust = manual_adjusted_scheme_2(current_epoch)
    #decay, grow, norm_clip_rate, write_rate_adjust = cifar_test_scheme(current_epoch)

    current_learning_rate = ilr * decay / accum_grad_num
    # y bank loss rate no need to divide by accum_grad_sum
    # since it has been included in learning rate
    cybl1lr = fybl1lr * grow

    current_norm_clip = max_norm_clip * norm_clip_rate

    current_write_rate = final_write_rate * write_rate_adjust

    #current_learning_rate = initial_learning_rate * (0.5 ** (current_epoch // 10))
    # cast the generator into list
    all_params = list(model.parameters())

    print("total num of params: " + str(len(all_params)))

    # setup SGD algorithm
    optimizer = optim.SGD(all_params, lr=current_learning_rate,
                          momentum=0.9, weight_decay=weight_decay, nesterov=True)

    print("current learning rate:")
    print(current_learning_rate)

    print("current write rate:")
    print(current_write_rate)

    print("current y bank L1 loss rate")
    print(cybl1lr)

    optimizer.zero_grad()
    running_loss = 0.0
    acc_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # retrieve data
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        error_loss = criterion(outputs, labels)

        if ('memorize' in model.__class__.__dict__):
            reg_loss = model.L1_bank_y_reg(cybl1lr)
            loss = error_loss + reg_loss
        else:
            loss = error_loss

        acc_loss += loss.item()

        # if found nan in loss, clear the losses
        if check_mat_nan(loss) or np.isnan(acc_loss):
            acc_loss = 0.0
            optimizer.zero_grad()
            print("observe nan, skip the cycle")
        else:
            # calculate backward everytime
            if backward_all:
                loss.backward()
                if (i + 1) % accum_grad_num == 0:
                    # clip the gradient
                    if norm_clip_option:
                        torch.nn.utils.clip_grad_norm_(params_with_grad, current_norm_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            # use total loss instead
            else:
                if (i + 1) % accum_grad_num == 0:
                    acc_loss.backward()
                    # clip the gradient
                    if norm_clip_option:
                        torch.nn.utils.clip_grad_norm_(params_with_grad, current_norm_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    acc_loss = 0.0

            # turn on the bank normalization
            if ('memory_bank_normalize' in model.__class__.__dict__) and bndt:
                model.memory_bank_normalize()

            running_loss += loss.item()
            if i % loss_interval == 0:
                print('[%d, %5d] loss: %.3f' %
                      (current_epoch + 1, i + 1, running_loss / loss_interval))
                running_loss = 0.0

            # if class is an NTM, memorize the data point
            if ('memorize' in model.__class__.__dict__):
                model.memorize(inputs, labels, current_write_rate)

    end = time.time()
    print("this epoch takes a total of: " + str(end - start) + " seconds \n")


def training(model, total_epoch, ilr,
             trainloader, testloader, if_ten_crop=False,
             test_device='cpu', loss_interval=50, bndt=False,
             accum_grad_num=1, backward_all=True,
             fname='pytorch_model',
             norm_clip_option=False, forget_option=False,
             forget_factor=0.98, forget_amount=2,
             forget_epoch=80, forget_interval=5,
             final_write_rate=1e-3, fybl1lr=1e-3,
             test_interval=1):
    print("weight_decay")
    print(weight_decay)
    print("final_write_rate")
    print(final_write_rate)
    print("bank_L1_reg_rate")
    print(fybl1lr)

    best_acc = 0
    best_epoch = 0

    criterion = nn.CrossEntropyLoss()

    # guard nan in the model
    model.register_backward_hook(gradient_nan_guard_hook)

    for epoch in range(total_epoch):
        train_action(model, epoch, ilr, criterion,
                     trainloader, testloader, if_ten_crop,
                     test_device, loss_interval, bndt,
                     accum_grad_num, backward_all,
                     norm_clip_option, fname,
                     final_write_rate, fybl1lr)

        # forget conditions
        # model is NTM ?
        is_NTM_model = ('memorize' in model.__class__.__dict__)
        # reach the epoch ?
        forget_epoch_reached = (epoch >= forget_epoch)
        # reach the interval ?
        forget_interval_reached = ((epoch - forget_epoch) % forget_interval == 0)
        # forget or not?
        if forget_option and is_NTM_model and forget_epoch_reached and forget_interval_reached:
            model.forget(forget_factor, forget_amount)

        if (epoch + 1) % test_interval == 0:
            # calc acc
            # calc train acc
            _, _ = get_acc(model, trainloader, if_ten_crop, test_device, set_name="train")
            # calc test acc
            acc, similarity_collection = get_acc(model, testloader,
                                                 if_ten_crop, test_device, set_name="test")

            if ('memorize' in model.__class__.__dict__):
                key_bank_status = model.memory_bank.key_bank.data
                y_bank_status = model.memory_bank.y_bank.data

                usage_status = model.memory_bank.usage.data
                usage_status = usage_status.view(-1, 1)

                status = torch.cat([y_bank_status, usage_status], dim=1)

                print("current y bank and usage status:")
                print(status)

            acc = max(acc)

            # compare for best model
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

                # save model if it is the best
                torch.save(model, fname + '.pt')

                if ('memorize' in model.__class__.__dict__):
                    np.save("key_bank_status.npy", np.array(key_bank_status))
                    np.save("y_bank_status.npy", np.array(status))
                    np.save("similarity_collection.npy", np.array(similarity_collection))

            print("currently best performance: " + str(best_acc) +
                  "%" + " at epoch: " + str(best_epoch))

            #time.sleep(60 * test_interval)

    print('Finished Training')
