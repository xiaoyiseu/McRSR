import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

#**************************************************
#                       MCU Loss 
#************************************************** 
class CentroidManager:
    def __init__(self, momentum=0.9, logit_scale=8, epsilon=1e-8):
        self.img_centroids = {}
        self.txt_centroids = {}
        self.momentum = momentum
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logit_scale = logit_scale

    def update_centroids(self, features, labels, feature_type='img'):
        centroids = self.img_centroids if feature_type == 'img' else self.txt_centroids
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_features = features[label_mask]
            centroid = label_features.mean(dim=0)
            if label in centroids:
                centroids[label] = self.momentum * centroids[label] + (1 - self.momentum) * centroid
            else:
                centroids[label] = centroid

        if feature_type == 'img':
            self.img_centroids = centroids
        else:
            self.txt_centroids = centroids

    def cross_modal_prob(self, img_centroids, txt_centroids):
        img_centroid_list = torch.stack(list(img_centroids.values()))
        txt_centroid_list = torch.stack(list(txt_centroids.values()))
        dist = torch.cdist(img_centroid_list, txt_centroid_list, p=2).pow(2)
        dist = dist - dist.max(dim=1, keepdim=True).values
        probabilities = F.softmax((-dist * self.logit_scale) + self.epsilon, dim=1)
        return probabilities

    def centroid_contrastive_loss(self, img_features, txt_features, labels):
        img_features, txt_features, labels = img_features.to(self.device), txt_features.to(self.device), labels.to(self.device)
        
        self.update_centroids(img_features, labels, feature_type='img')
        self.update_centroids(txt_features, labels, feature_type='txt')
        
        cross_modal_probs = self.cross_modal_prob(self.img_centroids, self.txt_centroids).clamp(min=self.epsilon)

        img_labels = torch.tensor(list(self.img_centroids.keys()), device=self.device)
        txt_labels = torch.tensor(list(self.txt_centroids.keys()), device=self.device)
        
        cross_modal_mask = (img_labels.unsqueeze(1) == txt_labels.unsqueeze(0)).float().clamp(min=self.epsilon)
        target_probs = cross_modal_mask / (cross_modal_mask.sum(dim=1, keepdim=True))

        cross_modal_loss = F.kl_div(cross_modal_probs.log(), target_probs, reduction='batchmean')

        return cross_modal_loss

def MCU_loss(features_img, features_txt, labels, momentum=0.5):
    logit_scale = 8
    features_img = features_img / features_img.norm(dim=1, keepdim=True)
    features_txt = features_txt / features_txt.norm(dim=1, keepdim=True)   
    centroid_manager = CentroidManager(logit_scale = logit_scale, momentum = momentum)
    loss = centroid_manager.centroid_contrastive_loss(features_img, features_txt, labels) 
    return loss

#**************************************************
#                       PPE Loss 
#************************************************** 
class NaiveBayesAnalysis(Module):
    def __init__(self, device='cuda', logit_scale=50., contrastive_lambda=1e-3, alpha = 0.1, beta=2.):
        super(NaiveBayesAnalysis, self).__init__()
        self.device = device
        self.logit_scale = logit_scale
        self.contrastive_lambda = contrastive_lambda
        self.alpha = alpha
        self.beta = beta
    
    def NormPart(self, P_x_given_y, mask):      
        P_x_given_y_expanded = P_x_given_y.unsqueeze(1)  # (batch_size, 1, num_classes)
        pairwise_diffs = P_x_given_y_expanded - P_x_given_y_expanded.transpose(1, 2)
        pairwise_distances = pairwise_diffs.norm(dim=0)
        contrastive_reg = (pairwise_distances**2 * mask).sum()
        contrastive_reg *= self.contrastive_lambda
        return contrastive_reg

    def forward(self, P, Q, y_labels):       
        unique_y = y_labels.unique(sorted=True)
        num_classes = len(unique_y)
        class_to_idx = {cls.item(): idx for idx, cls in enumerate(unique_y)}
        mapped_labels = torch.tensor([class_to_idx[y.item()] for y in y_labels], dtype=torch.long).to(y_labels.device)
        similarity_matrix = F.cosine_similarity(P.unsqueeze(1), Q.unsqueeze(0), dim=2)

        similarity_matrix_exp = torch.exp(similarity_matrix * self.logit_scale)
        similarity_matrix_prob = similarity_matrix_exp / similarity_matrix_exp.sum(dim=1, keepdim=True)  # 归一化为概率

        counts_y = mapped_labels.bincount(minlength=num_classes)

        P_y = (counts_y.float() + self.alpha) / (len(y_labels) + self.alpha * num_classes)

        P_x1_given_y = torch.zeros(P.size(0), num_classes).to(self.device)
        P_x2_given_y = torch.zeros(P.size(0), num_classes).to(self.device)
        P_x1_x2 = torch.zeros(P.size(0), num_classes).to(self.device)

        for i, y in enumerate(unique_y):
            indices = (mapped_labels == class_to_idx[y.item()]).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                # P(x1|y) and P(x2|y)
                P_x1_given_y[:, i] = similarity_matrix_prob[:, indices].mean(dim=1)
                P_x2_given_y[:, i] = similarity_matrix_prob[indices, :].mean(dim=0)
                # P(x1, x2) 
                P_x1_x2[:, i] = similarity_matrix_prob[indices, :][:, indices].mean()
            else:
                P_x1_given_y[:, i] = 1e-10
                P_x2_given_y[:, i] = 1e-10
                P_x1_x2[:, i] = 1e-10
        P_x1_x2 = torch.sum(P_x1_x2 * P_y.unsqueeze(0), dim=1).clamp(min=1e-10)

        # P(y|x1, x2)
        P_y = P_y.to(self.device)
        P_x1_given_y = P_x1_given_y.to(self.device)
        P_x2_given_y = P_x2_given_y.to(self.device)

        P_y_given_x1_x2_numer = P_y.unsqueeze(0) * P_x1_given_y * P_x2_given_y
        P_y_given_x1_x2 = P_y_given_x1_x2_numer / (P_x1_x2.unsqueeze(1))
        P_y_given_x1_x2 = torch.clamp(P_y_given_x1_x2, min=1e-10)
        P_y_given_x1_x2 /= P_y_given_x1_x2.sum(dim=1, keepdim=True)

        target_one_hot = F.one_hot(mapped_labels, num_classes).float().to(self.device)
        log_probs = torch.log(P_y_given_x1_x2)
        cce_loss = -torch.sum(target_one_hot * log_probs) / P_y_given_x1_x2.size(0)

        mask = 1 - torch.eye(num_classes, device=self.device)
        contrastive_reg1 = self.NormPart(P_x1_given_y, mask)
        contrastive_reg2 = self.NormPart(P_x2_given_y, mask)
        total_loss = cce_loss + contrastive_reg1 + contrastive_reg2

        return total_loss

def PPE_loss(image_features, text_features, labels):
    logit_scale = 5
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    criterion = NaiveBayesAnalysis(logit_scale = logit_scale)

    loss1 = criterion(image_features, text_features, labels)
    loss2 = criterion(text_features, image_features, labels)
    loss = loss1 + loss2
    
    return loss

#**************************************************
#                       FSA Loss 
#************************************************** 
def FSA_loss(img_features, txt_features, labels):
    logit_scale = 55
    img_norm = img_features / img_features.norm(dim=1, keepdim=True)
    txt_norm = txt_features / txt_features.norm(dim=1, keepdim=True)
    
    similarity_matrix = img_norm @ txt_norm.t()
    scaled_matrix = torch.relu(similarity_matrix)

    A = (torch.trace(similarity_matrix)/img_norm.shape[0]).detach()   
    F = (torch.norm(similarity_matrix, p='fro')/img_norm.shape[0]).detach()
    logit_scale = logit_scale / (1.0 + torch.exp(A+F))
    
    labels_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

    pos_sim = scaled_matrix[labels_mask]
    logits = scaled_matrix * logit_scale
    pos_sim_exp = torch.exp(pos_sim * logit_scale)

    logits_exp = torch.exp(logits)
    logits_sum = logits_exp.sum(dim=1, keepdim=True)
    pos_loss = -torch.log(pos_sim_exp / logits_sum.clamp(min=1e-8))
    loss = pos_loss.mean()
    return loss 