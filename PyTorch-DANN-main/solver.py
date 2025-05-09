import torch
import torch.nn as nn
import os
from torch import optim
from model import encoder, classifier, discriminator
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from utils import adjust_alpha, adjust_alpha_log_growth, replace_nan_inf_hook
from data_loader import get_loader
import copy
import torch.nn.functional as F
from torchsummary import summary

class Solver(object):
    def __init__(self, args):
        self.args = args

        self.s_train_loader, self.s_test_loader, self.t_train_loader, self.t_test_loader = get_loader(args)

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        self.best_acc = 0
        self.time_taken = None

        self.enc = encoder(self.args).cuda()
        self.clf = classifier(self.args).cuda()
        self.fd = discriminator(self.args).cuda()
        

        print('--------Network--------')
        print(self.enc)
        print(self.clf)

        print('--------Feature Disc--------')
        print(self.fd)

        self.fake_label = torch.FloatTensor(self.args.batch_size, 1).fill_(0).cuda()
        self.real_label = torch.FloatTensor(self.args.batch_size, 1).fill_(1).cuda()
        #self.error_testing_lb = torch.FloatTensor(self.args.batch_size, 1).fill_(5).cuda()
        
        if not args.method == 'src':
            if os.path.exists(os.path.join(self.args.model_path, 'src_enc.pt')):
                print("Loading Source model...")
                self.enc.load_state_dict(torch.load(os.path.join(self.args.model_path, 'src_enc.pt')))
                self.clf.load_state_dict(torch.load(os.path.join(self.args.model_path, 'src_clf.pt')))
            else:
                print("Training Source model...")
                self.src()
                self.test()

        if self.args.mode == 'test':
            if os.path.exists(os.path.join(self.args.model_path, 'dann_enc.pt')):
                print("Loading DANN model")
                self.enc.load_state_dict(torch.load(os.path.join(self.args.model_path,'dann_enc.pt')))
                self.clf.load_state_dict(torch.load(os.path.join(self.args.model_path,'dann_clf.pt')))

    def log_gradient_norms(self, model):
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()  # L2 norm
                print(f'Layer: {name}, Gradient Norm: {param_norm}')
                total_norm += param_norm ** 2
        total_norm = total_norm
        print(f'{model.name} Gradient Norm Squared: {total_norm}')

#Functions for accuracy and mcc testing

    def test_dataset(self, db='t_test'):
        self.enc.eval()
        self.clf.eval()

        actual = []
        pred = []

        if db.lower() == 's_train':
            loader = self.s_train_loader
        elif db.lower() == 's_test':
            loader = self.s_test_loader
        elif db.lower() == 't_train':
            loader = self.t_train_loader
        else:
            loader = self.t_test_loader

        for data in loader:
            img, label = data

            img = img.cuda()

            with torch.no_grad():
                class_out = self.clf(self.enc(img))
            _, predicted = torch.max(class_out.data, 1)

            actual += label.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        mcc = matthews_corrcoef(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.num_classes))

        return acc, cm, mcc

    def test(self):
        s_train_acc, cm, mcc = self.test_dataset('s_train')
        print("Source Tr Acc: %.2f" % (s_train_acc))
        print("Source Tr Mcc: %.2f" % (mcc))
        if self.args.cm:
            print(cm)

        s_test_acc, cm, mcc = self.test_dataset('s_test')
        print("Source Te Acc: %.2f" % (s_test_acc))
        print("Source Te Mcc: %.2f" % (mcc))
        if self.args.cm:
            print(cm)

        t_train_acc, cm, mcc = self.test_dataset('t_train')
        print("Target Tr Acc: %.2f" % (t_train_acc))
        print("Target Tr Mcc: %.2f" % (mcc))
        if self.args.cm:
            print(cm)

        t_test_acc, cm, mcc = self.test_dataset('t_test')
        print("Target Te Acc: %.2f" % (t_test_acc))
        print("Target Te Mcc: %.2f" % (mcc))
        if self.args.cm:
            print(cm)

        return s_train_acc, s_test_acc, t_train_acc, t_test_acc

#This function trains the source-only model, so that the feature extractor and the classifier are pre-trained on the source domain

    def src(self):
        total_iters = 0
        self.best_acc = 0
        s_iter_per_epoch = len(iter(self.s_train_loader))
        self.args.src_test_epoch = max(self.args.src_epochs // 10, 1)

        self.optimizer = optim.Adam(list(self.enc.parameters()) + list(self.clf.parameters()), self.args.lr,
                                    betas=[0.5, 0.999], weight_decay=self.args.weight_decay)

        for epoch in range(self.args.src_epochs):

            self.clf.train()
            self.enc.train()

            for i, (source, s_labels) in enumerate(self.s_train_loader):
                total_iters += 1

                source, s_labels = source.cuda(), s_labels.cuda()

                s_logits = self.clf(self.enc(source))
                s_clf_loss = self.ce(s_logits, s_labels)
                loss = s_clf_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 50 == 0 or i == (s_iter_per_epoch - 1):
                    print('Ep: %d/%d, iter: %d/%d, total_iters: %d, s_err: %.4f'
                          % (epoch + 1, self.args.src_epochs, i + 1, s_iter_per_epoch, total_iters, s_clf_loss))

            if (epoch + 1) % self.args.src_test_epoch == 0:
                s_test_acc, cm = self.test_dataset('s_test')
                print("Source test acc: %0.2f" % (s_test_acc))
                if self.args.cm:
                    print(cm)

                if s_test_acc > self.best_acc:
                    self.best_acc = s_test_acc
                    best_enc = copy.deepcopy(self.enc.state_dict())
                    best_clf = copy.deepcopy(self.clf.state_dict())

        torch.save(best_enc, os.path.join(self.args.model_path, 'src_enc.pt'))
        torch.save(best_clf, os.path.join(self.args.model_path, 'src_clf.pt'))

        self.enc.load_state_dict(best_enc)
        self.clf.load_state_dict(best_clf)


#This function does the adversarial training on the dann model, in the following way: the pictures from the source and the target domain are sent in the feature extractor, the features are then passed on to the classifier and the domain discriminator. The classifier continues to learn the labels of the images of the source domain, while the domain discriminator has a binnary output 0 for the target domain 1 for the source domain.

    def dann(self):

        s_iter_per_epoch = len(self.s_train_loader)
        t_iter_per_epoch = len(self.t_train_loader)
        min_len = min(s_iter_per_epoch, t_iter_per_epoch)
        total_iters = 0

        print("Source iters per epoch: %d" % (s_iter_per_epoch))
        print("Target iters per epoch: %d" % (t_iter_per_epoch))
        print("iters per epoch: %d" % (min(s_iter_per_epoch, t_iter_per_epoch)))        


        self.optimizer = optim.Adam(list(self.enc.parameters()) + list(self.clf.parameters()) + list(self.fd.parameters()), self.args.lr,
                                      betas=[0.5, 0.999], weight_decay=self.args.weight_decay)

        for epoch in range(self.args.adapt_epochs):
            self.clf.train()
            self.enc.train()
            self.fd.train()
            
            

            for i, (source_data, target_data) in enumerate(zip(self.s_train_loader, self.t_train_loader)):
                total_iters += 1
                
                #This parameter regulate the magnitude of the reversed gradient passed to the feature extractor
                alpha = adjust_alpha_log_growth(i, epoch, min_len, self.args.adapt_epochs,self.args.num_branches)

                    
                source, s_labels = source_data
                source, s_labels = source.cuda(), s_labels.cuda()

                target, t_labels = target_data
                target, t_labels = target.cuda(), t_labels.cuda()

                s_deep = self.enc(source)
                s_out = self.clf(s_deep)


                t_deep = self.enc(target)
                #t_out = self.clf(t_deep)

                
                s_fd_out = self.fd(s_deep, alpha=alpha)
                check_tensor=torch.isfinite(s_fd_out)
                for e in check_tensor:
                    if e==False:
                        print(check_tensor)
                        print("trovato nan o inf in etichette source")
                        return
                t_fd_out = self.fd(t_deep, alpha=alpha)
                check_tensor=torch.isfinite(t_fd_out)
                for e in check_tensor:
                    if e==False:
                        print(check_tensor)
                        print("trovato nan o inf in etichette target")
                        return 

                
                s_domain_err = self.bce(s_fd_out, self.real_label)                
                t_domain_err = self.bce(t_fd_out, self.fake_label)
                disc_loss = (s_domain_err + t_domain_err)/2

                s_clf_loss = self.ce(s_out, s_labels)

                loss = s_clf_loss + disc_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fd.parameters(), max_norm=5.0)               
                self.optimizer.step()

                if i % 50 == 0 or i == (min_len - 1):
                    print('Ep: %d/%d, iter: %d/%d, total_iters: %d, s_err: %.4f, d_err: %.4f, alpha: %.4f'
                          % (epoch + 1, self.args.adapt_epochs, i + 1, min_len, total_iters, s_clf_loss, disc_loss, alpha))
            
            print(f'Loss={loss}')
            print(f'Loss ratio={disc_loss/s_clf_loss}')
            if epoch % 5 == 0:
                t_test_acc, cm, t_target_mcc = self.test_dataset('t_test')
                print("Target test acc: %0.2f" % (t_test_acc))
                print("Target test mcc: %0.2f" % (t_target_mcc))
                if self.args.cm:
                    print(cm)
                torch.save(self.enc.state_dict(), os.path.join(self.args.model_path, 'dann_enc.pt'))
                torch.save(self.clf.state_dict(), os.path.join(self.args.model_path, 'dann_clf.pt'))
                torch.save(self.fd.state_dict(), os.path.join(self.args.model_path, 'dann_disc.pt'))        
