# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
import shap
import utilities.data_gen_utils
from shap_library_utils.explainers._manifoldshap import ManifoldShap, ManifoldShapRejectionSampling
from shap_library_utils.explainers._conditionalshap import ConditionalShap
from manifold_classifier.create_classifiers import get_classifier, ClassifierWrapper
from utilities.arguments import set_args
from utilities.models import MLPWrapper
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def main(args):
    """
    Here, we compare the results of ManifoldShap with other methodologies 
    with changing feature dimensions d.
    Input data follows the distribution
    X ~ N(0, A)
    where A_{ij} = 1 if i = j, and {corr} otherwise. 
    """

    results_dir = "../multivariate_experiments"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    d = 100
    corr = 0.9
    gt_model = lambda x: x[:, 0]

    # Generate Training and Testing Data
    logging.info("Generating Data")
    cov = np.ones((d, d))*corr
    np.fill_diagonal(cov, 0)
    cov += np.eye(d)

    train_data_X = np.random.multivariate_normal(np.zeros(d), cov, (args.n_train,))
    train_data_Y = gt_model(train_data_X)

    test_data_X = np.random.multivariate_normal(np.zeros(d), cov, (args.n_test,))
    test_data_Y = gt_model(test_data_X)

    # Train classifier for on-manifold data OR Use kernel/vae density estimation
    logging.info("Get the classifier model")
    density_model, manifold_threshold, _, _ = get_classifier(args, x_train=torch.FloatTensor(train_data_X),
                                                                   x_test=torch.FloatTensor(test_data_X),
                                                                   quantile_threshold=0.000001,
                                                                   return_binary_classifier=False, 
                                                                   visualise=False)
    manifold_classifier = ClassifierWrapper(density_model, manifold_threshold)

    # Density estimation for RJBShap
    def model_with_density(x):
        with torch.no_grad():
            return density_model(torch.FloatTensor(x)).cpu().numpy()*gt_model(x)

    # Define all shapley calculators and Save all shapley values
    #Setup IS
    logging.info("Running IS")
    background_distribution = shap.kmeans(train_data_X,10)
    interventional_shap_explainer = shap.KernelExplainer(gt_model, background_distribution)
    interventional_shap_values = interventional_shap_explainer.shap_values(test_data_X)
    pd.DataFrame(data=interventional_shap_values, columns=[f'col{i}' for i in range(1, d+1)]).to_csv(
            f"{results_dir}/interventional_shap_multiv_gaussian_d{d}_corr{corr}.csv", index=False)
    logging.info("IS calculation complete")

    # Setup RJBShap
    logging.info("Running RJBShap")
    rjbshap_explainer = shap.KernelExplainer(model_with_density, background_distribution)
    rjbshap_values = rjbshap_explainer.shap_values(test_data_X)
    pd.DataFrame(data=rjbshap_values, columns=[f'col{i}' for i in range(1, d+1)]).to_csv(
            f"{results_dir}/rjbshap_multiv_gaussian_d{d}_corr{corr}.csv", index=False)
    logging.info("RJBShap calculation complete")

    #Setup ManifoldSHAP
    logging.info("Running ManifoldSHAP")
    def manifold(x):
        with torch.no_grad():
            return manifold_classifier(torch.FloatTensor(x.copy())).cpu().detach().numpy()
    manifold_explainer = ManifoldShapRejectionSampling(
        gt_model, background_distribution, manifold)
    manifold_shap_values = manifold_explainer.shap_values(test_data_X)
    pd.DataFrame(data=manifold_shap_values, columns=[f'col{i}' for i in range(1, d+1)]).to_csv(
        f"{results_dir}/man_shap_multiv_gaussian_d{d}_corr{corr}.csv", index=False)
    logging.info("ManifoldSHAP calculation complete")

    # Setup ConditionalSHAP
    logging.info("Running ConditionalSHAP")
    cond_shap_explainer = ConditionalShap(
        gt_model, train_data_X, output_dim=1, use_softmax=False, class_shaps_to_output=0, max_iters=200)
    cond_shap_values = cond_shap_explainer.shap_values(test_data_X)
    pd.DataFrame(data=cond_shap_values, columns=[f'col{i}' for i in range(1, d+1)]).to_csv(
        f"{results_dir}/cond_shap_multiv_gaussian_d{d}_corr{corr}.csv", index=False)
    logging.info("ConditionalSHAP calculation complete")


if __name__ == "__main__":
    args = set_args()
    main(args)
