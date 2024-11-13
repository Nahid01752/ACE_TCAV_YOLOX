"""ACE library.

Library for discovering and testing concept activation vectors. It contains
ConceptDiscovery class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score..
"""
import json
import sys
import gc
import glob
from collections import OrderedDict
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import pandas as pd
from tcav import cav
from ace_helpers import *

# append to the root
yolox_path = '/Users//nahid007/PycharmProjects/globalXAI/YOLOX'
sys.path.append(yolox_path)

# import yolox architecture
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.models.network_blocks import CSPLayer, Bottleneck, BaseConv
from yolox.models.darknet import CSPDarknet

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class ConceptDiscovery(object):
    """Discovering and testing concepts of a class.

  For a trained network, it first discovers the concepts as areas of the iamges
  in the class and then calculates the TCAV score of each concept. It is also
  able to transform images from pixel space into concept space.
  """

    def __init__(self,
                 model,
                 target_class,
                 random_concept,
                 bottlenecks,
                 head,
                 source_dir,
                 activation_dir,
                 cav_dir,
                 num_random_exp=2,
                 channel_mean=True,
                 max_imgs=40,
                 min_imgs=20,
                 num_discovery_imgs=40,
                 num_workers=0,
                 average_image_value=117,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """Runs concept discovery for a given class in a trained model.

    For a trained classification model, the ConceptDiscovery class first
    performs unsupervised concept discovery using examples of one of the classes
    in the network.

    Args:
      model: A trained classification model on which we run the concept
             discovery algorithm
      target_class: Name of the one of the classes of the network
      random_concept: A concept made of random images (used for statistical
                      test) e.g. "random500_199"
      bottlenecks: a list of bottleneck layers of the model for which the cocept
                   discovery stage is performed
      source_dir: This directory that contains folders with images of network's
                  classes.
      activation_dir: directory to save computed activations
      cav_dir: directory to save CAVs of discovered and random concepts
      num_random_exp: Number of random counterparts used for calculating several
                      CAVs and TCAVs for each concept (to make statistical
                        testing possible.)
      channel_mean: If true, for the unsupervised concept discovery the
                    bottleneck activations are averaged over channels instead
                    of using the whole acivation vector (reducing
                    dimensionality)
      max_imgs: maximum number of images in a discovered concept
      min_imgs : minimum number of images in a discovered concept for the
                 concept to be accepted
      num_discovery_imgs: Number of images used for concept discovery. If None,
                          will use max_imgs instead.
      num_workers: if greater than zero, runs methods in parallel with
        num_workers parallel threads. If 0, no method is run in parallel
        threads.
      average_image_value: The average value used for mean subtraction in the
                           nework's preprocessing stage.
    """
        self.model = model
        self.target_class = target_class
        self.num_random_exp = num_random_exp
        if isinstance(bottlenecks, str):
            bottlenecks = [bottlenecks]
        self.bottlenecks = bottlenecks
        self.head = head
        self.source_dir = source_dir
        self.activation_dir = activation_dir
        self.cav_dir = cav_dir
        self.channel_mean = channel_mean
        self.random_concept = random_concept
        self.image_shape = [224, 224]
        self.mean = mean
        self.std = std
        self.max_imgs = max_imgs
        self.min_imgs = min_imgs
        if num_discovery_imgs is None:
            num_discovery_imgs = max_imgs
        self.num_discovery_imgs = num_discovery_imgs
        self.num_workers = num_workers
        self.average_image_value = average_image_value

    def load_concept_imgs(self, concept, max_imgs=1000):
        """Loads all colored images of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    Returns:
      Images of the desired concept or class.
    """
        concept_dir = os.path.join(self.source_dir, concept)
        img_paths = glob.glob(concept_dir + '/*')
        return load_images_from_files(
            img_paths,
            max_imgs=max_imgs,
            return_filenames=False,
            do_shuffle=False,
            run_parallel=(self.num_workers > 0),
            shape=(self.image_shape),
            num_workers=self.num_workers)

    def create_patches(self, method='slic', discovery_images=None,
                       param_dict=None):
        """Creates a set of image patches using superpixel methods.

    This method takes in the concept discovery images and transforms it to a
    dataset made of the patches of those images.

    Args:
      method: The superpixel method used for creating image patches. One of
        'slic', 'watershed', 'quickshift', 'felzenszwalb'.
      discovery_images: Images used for creating patches. If None, the images in
        the target class folder are used.

      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    """
        if param_dict is None:
            param_dict = {}
        dataset, image_numbers, patches = [], [], []
        if discovery_images is None:
            raw_imgs = self.load_concept_imgs(
                self.target_class, self.num_discovery_imgs)
            self.discovery_images = raw_imgs
        else:
            self.discovery_images = discovery_images

        # check discovery images
        print(f"Number of discovery images: {len(self.discovery_images)}")

        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            outputs = pool.map(
                lambda img: self._return_superpixels(img, method, param_dict),
                self.discovery_images)
            for fn, sp_outputs in enumerate(outputs):
                image_superpixels, image_patches = sp_outputs
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(fn)
        else:
            for fn, img in enumerate(self.discovery_images):
                image_superpixels, image_patches = self._return_superpixels(
                    img, method, param_dict)
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(fn)
        self.dataset, self.image_numbers, self.patches = \
            np.array(dataset), np.array(image_numbers), np.array(patches)

        # Debug statements to check sizes
        print(f"Dataset size: {self.dataset.shape}")
        print(f"Image numbers size: {self.image_numbers.shape}")
        print(f"Patches size: {self.patches.shape}")

    def _return_superpixels(self, img, method='slic',
                            param_dict=None):
        """Returns all patches for one image.

    Given an image, calculates superpixels for each of the parameter lists in
    param_dict and returns a set of unique superpixels by
    removing duplicates. If two patches have Jaccard similarity more than 0.5,
    they are concidered duplicates.

    Args:
      img: The input image
      method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    Raises:
      ValueError: if the segementation method is invaled.
    """
        if param_dict is None:
            param_dict = {}
        if method == 'slic':
            n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
            n_params = len(n_segmentss)
            compactnesses = param_dict.pop('compactness', [20] * n_params)
            sigmas = param_dict.pop('sigma', [1.] * n_params)
        elif method == 'watershed':
            markerss = param_dict.pop('marker', [15, 50, 80])
            n_params = len(markerss)
            compactnesses = param_dict.pop('compactness', [0.] * n_params)
        elif method == 'quickshift':
            max_dists = param_dict.pop('max_dist', [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.pop('ratio', [1.0] * n_params)
            kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
        elif method == 'felzenszwalb':
            scales = param_dict.pop('scale', [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.pop('sigma', [0.8] * n_params)
            min_sizes = param_dict.pop('min_size', [20] * n_params)
        else:
            raise ValueError('Invalid superpixel method!')
        unique_masks = []
        for i in range(n_params):
            param_masks = []
            if method == 'slic':
                segments = segmentation.slic(
                    img, n_segments=n_segmentss[i], compactness=compactnesses[i],
                    sigma=sigmas[i])
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img, markers=markerss[i], compactness=compactnesses[i])
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                    ratio=ratios[i])
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)
        superpixels, patches = [], []
        while unique_masks:
            superpixel, patch = self._extract_patch(img, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)
        return superpixels, patches

    def _extract_patch(self, image, mask):
        """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
        mask_expanded = np.expand_dims(mask, -1)

        patch = (mask_expanded * image + (
                1 - mask_expanded) * float(self.average_image_value) / 255)  # superpixels

        ones = np.where(mask == 1)  # get the pixels where pixels belonging to the segment are 1, others == 0
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()

        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))

        image_resized = np.array(image.resize(self.image_shape,
                                              Image.BICUBIC)).astype(float) / 255

        return image_resized, patch

    def _patch_activations(self, imgs, bottleneck, bs=100, channel_mean=None, transform=None):
        """Returns activations of a list of imgs.

    Args:
      imgs: List/array of images to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """
        print("Starting _patch_activations")
        print(f"imgs shape: {imgs.shape}")
        print(f"bottleneck: {bottleneck}")
        print(f"batch size (bs): {bs}")
        # print(f"channel_mean: {channel_mean}")

        if channel_mean is None:
            channel_mean = self.channel_mean

        self.model.eval()

        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            output = pool.map(
                lambda i: self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck),
                np.arange(int(imgs.shape[0] / bs) + 1))
        else:
            output = []
            num_batches = int(np.ceil(imgs.shape[0] / bs))
            for i in range(num_batches):
                start_idx = i * bs
                end_idx = min((i + 1) * bs, imgs.shape[0])
                img_batch = imgs[start_idx:end_idx]

                img_batch_tensor = torchvision.transforms.functional.normalize(
                    torch.tensor(img_batch).permute(0, 3, 1, 2),
                    mean=self.mean, std=self.std).float()

                # clear feature blobs
                global features_blobs
                features_blobs.clear()
                # print(f"Before forward pass, features_blobs: {features_blobs}")
                # forward pass
                _ = self.model(img_batch_tensor)
                # print(f"After forward pass, features_blobs: {features_blobs}")
                #if bottleneck == 'backbone.backbone.stem':
                #   out_batch = features_blobs[0].transpose(0, 2, 3, 1)
                #elif bottleneck == 'backbone.backbone.dark2.0':
                #   out_batch = features_blobs[1].transpose(0, 2, 3, 1)

                #elif bottleneck == 'backbone.backbone.dark2':
                #   out_batch = features_blobs[2].transpose(0, 2, 3, 1)
                #elif bottleneck == 'backbone.backbone.dark3':
                #   out_batch = features_blobs[3].transpose(0, 2, 3, 1)
                #elif bottleneck == 'backbone.backbone.dark4':
                #   out_batch = features_blobs[4].transpose(0, 2, 3, 1)
                #elif bottleneck == 'backbone.backbone.dark5.1':
                #   out_batch = features_blobs[5].transpose(0, 2, 3, 1)

                #elif bottleneck == 'backbone.backbone.dark5':
                #   out_batch = features_blobs[6].transpose(0, 2, 3, 1)

                # if bottleneck == 'backbone.C3_p4':
                #   out_batch = features_blobs[0].transpose(0, 2, 3, 1)

                elif bottleneck == 'backbone.C3_p3':
                   out_batch = features_blobs[0].transpose(0, 2, 3, 1)

                elif bottleneck == 'backbone.C3_n3':
                   out_batch = features_blobs[1].transpose(0, 2, 3, 1)

                elif bottleneck == 'backbone.C3_n4':
                   out_batch = features_blobs[2].transpose(0, 2, 3, 1)

                #elif bottleneck == 'head.cls_convs.0':
                #   out_batch = features_blobs[11].transpose(0, 2, 3, 1)

                else:
                   out_batch = features_blobs[0].transpose(0, 2, 3, 1)


                # Check the state of features_blobs after the forward pass
                # if not features_blobs:
                #    print("Error: features_blobs is empty after model forward pass.")
                #    raise RuntimeError("features_blobs is empty. Ensure hooks are correctly implemented.")

                # for idx, feature_blob in enumerate(features_blobs):
                #    print(f"features_blobs[{idx}] shape after model forward pass: {feature_blob.shape}")

                # out_batch = features_blobs[0].transpose(0, 2, 3, 1)
                #try:
                #    out_batch = features_blobs[0].transpose(0, 2, 3, 1)
                    # print(f"out_batch shape: {out_batch.shape}")
                #except IndexError as e:
                #    print(f"Error accessing features_blobs[0]: {e}")
                #    raise

                features_blobs.pop()
                output.append(out_batch)

        output = np.concatenate(output, 0)
        print(f"Concatenated output shape: {output.shape}")

        if channel_mean and len(output.shape) > 3:
            output = np.mean(output, (1, 2))
        else:
            output = np.reshape(output, [output.shape[0], -1])
        print("Finished _patch_activations")
        return output

    def _cluster(self, acts, method='KM', param_dict=None):
        """Runs unsupervised clustering algorithm on concept actiavtations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
        print("Starting _cluster")
        if param_dict is None:
            param_dict = {}
        centers = None
        if method == 'KM':
            n_clusters = param_dict.pop('n_clusters', 25)
            print(f"Using KMeans with {n_clusters} clusters")
            km = cluster.KMeans(n_clusters)
            d = km.fit(acts)
            centers = km.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'AP':
            damping = param_dict.pop('damping', 0.5)
            ca = cluster.AffinityPropagation(damping)
            ca.fit(acts)
            centers = ca.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'MS':
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            asg = ms.fit_predict(acts)
        elif method == 'SC':
            n_clusters = param_dict.pop('n_clusters', 25)
            sc = cluster.SpectralClustering(
                n_clusters=n_clusters, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        elif method == 'DB':
            eps = param_dict.pop('eps', 0.5)
            min_samples = param_dict.pop('min_samples', 20)
            sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        else:
            raise ValueError('Invalid Clustering Method!')
        if centers is None:  # If clustering returned cluster centers, use medoids
            print("No centers returned, calculating medoids")
            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))
            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]
                print(f"Processing cluster {cluster_label}, number of points: {len(cluster_points)}")
                pw_distances = metrics.euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(
                    np.sum(pw_distances, -1))]
                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1)
                print(f"Medoid centers shape: {centers.shape}")
        else:
            print(f"Centers shape: {centers.shape}, Cost shape: {cost.shape}")
        print(f"Assignments (asg) shape: {asg.shape}")
        print("Finished _clustering")
        return asg, cost, centers

    def discover_concepts(self,
                          method='KM',
                          activations=None,
                          param_dicts=None):
        """Discovers the frequent occurring concepts in the target class.

      Calculates self.dic, a dicationary containing all the informations of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept, 'patches': original patches of the
      concepts, 'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
    """
        if param_dicts is None:
            param_dicts = {}
        if set(param_dicts.keys()) != set(self.bottlenecks):
            param_dicts = {bn: param_dicts for bn in self.bottlenecks}
        self.dic = {}   # The main dictionary of the ConceptDiscovery class.
        for bn in self.bottlenecks:
            print(f"Processing bottleneck: {bn}")
            bn_dic = {}
            if activations is None or bn not in activations.keys():
                print(f"No activations provided for bottleneck {bn}, computing patch activations.")
                bn_activations = self._patch_activations(self.dataset, bn)
            else:
                print(f"Using provided activations for bottleneck {bn}")
                bn_activations = activations[bn]

            # Print the shape of bn_activations
            print(f"Shape of bn_activations for bottleneck {bn}: {bn_activations.shape}")

            bn_dic['label'], bn_dic['cost'], centers = self._cluster(
                bn_activations, method, param_dicts[bn])
            print(f"Cluster labels shape: {bn_dic['label'].shape}")
            print(f"Cluster costs shape: {bn_dic['cost'].shape}")
            print(f"Cluster centers shape: {centers.shape}")

            concept_number, bn_dic['concepts'] = 0, []
            for i in range(bn_dic['label'].max() + 1):
                label_idxs = np.where(bn_dic['label'] == i)[0]
                if len(label_idxs) > self.min_imgs:
                    concept_costs = bn_dic['cost'][label_idxs]
                    concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]]
                    concept_image_numbers = set(self.image_numbers[label_idxs])
                    discovery_size = len(self.discovery_images)
                    highly_common_concept = len(
                        concept_image_numbers) > 0.5 * len(label_idxs)
                    mildly_common_concept = len(
                        concept_image_numbers) > 0.25 * len(label_idxs)
                    mildly_populated_concept = len(
                        concept_image_numbers) > 0.25 * discovery_size
                    cond2 = mildly_populated_concept and mildly_common_concept
                    non_common_concept = len(
                        concept_image_numbers) > 0.1 * len(label_idxs)
                    highly_populated_concept = len(
                        concept_image_numbers) > 0.5 * discovery_size
                    cond3 = non_common_concept and highly_populated_concept
                    if highly_common_concept or cond2 or cond3:
                        concept_number += 1
                        concept = '{}_concept{}'.format(self.target_class, concept_number)

                        bn_dic['concepts'].append(concept)
                        bn_dic[concept] = {
                            'images': self.dataset[concept_idxs],
                            'patches': self.patches[concept_idxs],
                            'image_numbers': self.image_numbers[concept_idxs]
                        }
                        bn_dic[concept + '_center'] = centers[i]
            bn_dic.pop('label', None)
            bn_dic.pop('cost', None)
            self.dic[bn] = bn_dic

    def _random_concept_activations(self, bottleneck, random_concept):
        """Wrapper for computing or loading activations of random concepts.

    Takes care of making, caching (if desired) and loading activations.

    Args:
      bottleneck: The bottleneck layer name
      random_concept: Name of the random concept e.g. "random500_0"

    Returns:
      A nested dict in the form of {concept:{bottleneck:activation}}
    """

        rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}'.format(
            random_concept, bottleneck)) + '.npy'

        if not os.path.exists(rnd_acts_path):
            rnd_imgs = self.load_concept_imgs(random_concept, self.max_imgs)

            acts = self.get_acts_from_images(rnd_imgs, bottleneck)

            np.save(rnd_acts_path, acts, allow_pickle=False)

            del acts
            del rnd_imgs

        activations = np.load(rnd_acts_path).squeeze()

        return activations

    def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
        """Calculates a single cav for a concept and a one random counterpart.

    Args:
      c: conept name
      r: random concept name
      bn: the layer name
      act_c: activation matrix of the concept in the 'bn' layer
      ow: overwrite if CAV already exists
      directory: to save the generated CAV

    Returns:
      The accuracy of the CAV
    """

        if directory is None:
            directory = self.cav_dir

        act_r = self._random_concept_activations(bn, r)

        cav_instance = cav.get_or_train_cav([c, r],
                                            bn, {
                                                c: {
                                                    bn: act_c
                                                },
                                                r: {
                                                    bn: act_r
                                                }
                                            },
                                            cav_dir=directory,
                                            overwrite=ow)

        accuracy = cav_instance.accuracies['overall']
        return accuracy

    def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
        """Calculates CAVs of a concept versus all the random counterparts.

    Args:
      bn: bottleneck layer name
      concept: the concept name
      activations: activations of the concept in the bottleneck layer
      randoms: None if the class random concepts are going to be used
      ow: If true, overwrites the existing CAVs

    Returns:
      A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
    """
        if randoms is None:
            randoms = [
                'random500_{}'.format(i) for i in np.arange(self.num_random_exp)
            ]
        if self.num_workers:
            pool = multiprocessing.Pool(20)
            accs = pool.map(
                lambda rnd: self._calculate_cav(concept, rnd, bn, activations, ow),
                randoms)
        else:
            accs = []
            for rnd in randoms:
                accuracy = self._calculate_cav(concept, rnd, bn, activations, ow)
                accs.append(accuracy)
        return accs

    def cavs(self, min_acc=0., ow=True):
        """Calculates cavs for all discovered concepts.

    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers

    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.

    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
    """
        print("Starting CAV calculations...")
        acc = {bn: {} for bn in self.bottlenecks}
        concepts_to_delete = []
        for bn in self.bottlenecks:
            print(f"Processing bottleneck: {bn}")

            for concept in self.dic[bn]['concepts']:
                # print(f"  Processing concept: {concept}")
                concept_imgs = self.dic[bn][concept]['images']
                concept_acts = self.get_acts_from_images(
                    concept_imgs, bn)
                acc[bn][concept] = self._concept_cavs(bn, concept, concept_acts, ow=ow)
                mean_acc = np.mean(acc[bn][concept])
                if mean_acc < min_acc:
                    concepts_to_delete.append((bn, concept))
            target_class_acts = self.get_acts_from_images(
                self.discovery_images, bn)

            acc[bn][self.target_class] = self._concept_cavs(
                bn, self.target_class, target_class_acts, ow=ow)

            rnd_acts = self._random_concept_activations(bn, self.random_concept)

            acc[bn][self.random_concept] = self._concept_cavs(
                bn, self.random_concept, rnd_acts, ow=ow)

        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)
        print("CAV calculations completed.")
        return acc

    def load_cav_direction(self, c, r, bn, directory=None):
        """Loads an already computed cav.
    Args:
      c: concept name
      r: random concept name
      bn: bottleneck layer
      directory: where CAV is saved

    Returns:
      The cav instance
    """
        if directory is None:
            directory = self.cav_dir
        params = {'model_type': 'linear', 'alpha': .01}
        cav_key = cav.CAV.cav_key([c, r], bn, params['model_type'], params['alpha'])
        cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
        vector = cav.CAV.load_cav(cav_path).cavs[0]
        return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

    def _sort_concepts(self, scores):
        for bn in self.bottlenecks:
            tcavs = []
            for concept in self.dic[bn]['concepts']:
                tcavs.append(np.mean(scores[bn][concept]))
            concepts = []
            for idx in np.argsort(tcavs)[::-1]:
                concepts.append(self.dic[bn]['concepts'][idx])
            self.dic[bn]['concepts'] = concepts

    def _return_gradients(self, images):
        """For the given images calculates the gradient tensors.

    Args:
      images: Images for which we want to calculate gradients.

    Returns:
      A dictionary of images gradients in all bottleneck layers.
    """
        print(f"Computing gradients...")
        gradients = {}
        class_id = self.model.label_to_id[self.target_class.replace('_', ' ')]
        for bn in self.bottlenecks:
            acts = self.get_acts_from_images(images, bn)
            bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
            for i in range(len(acts)):
                bn_grads[i] = self._get_gradients(acts[i:i + 1], [class_id], bn, example=None).reshape(-1)
            gradients[bn] = bn_grads
        print(f"Finished gradients computation.")
        return gradients

    def _debug_hook(self, name):
        def hook(module, input, output):
            print(f"Layer: {name}, Input shape: {input[0].shape}, Output shape: {output.shape}")

        return hook

    def _get_cutted_model(self, bottleneck):

        def traverse_and_collect(model, bottleneck_name):
            new_model_list = OrderedDict()
            add_to_list = False
            add_dark = True
            add_CP = True
            add_bu_conv1 = False
            add_bu_conv2 = False

            def _collect(mod, parent_name=''):
                nonlocal add_to_list, add_dark, add_CP, add_bu_conv1, add_bu_conv2
                for name, layer in mod.named_children():
                    full_name = parent_name + ('.' if parent_name else '') + name
                    sanitized_name = full_name.replace('.', '_')

                    if 'backbone.backbone.stem' == full_name and 'backbone.backbone.dark2' != bottleneck_name:
                        if add_to_list:         # 12 * 32
                            new_model_list[sanitized_name] = layer


                    if 'backbone.backbone.dark2.0' == full_name and 'backbone.backbone.dark2' != bottleneck_name:
                        if add_to_list:         # 32 * 64
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark2.1.conv3' == full_name and 'backbone.backbone.dark2' != bottleneck_name:
                        if add_to_list:         # 64 * 64
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark3.0' == full_name and 'backbone.backbone.dark3' != bottleneck_name:
                        if add_to_list:         # 64 * 128
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark3.1.conv3' == full_name  and 'backbone.backbone.dark3' != bottleneck_name:
                        if add_to_list:         # 128 * 128
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark4.0' == full_name and 'backbone.backbone.dark4' != bottleneck_name:
                        if add_to_list:         # 128 * 256
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark4.1.conv3' == full_name and 'backbone.backbone.dark4' != bottleneck_name:
                        if add_to_list:         # 256 * 256
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark5.0' == full_name and 'backbone.backbone.dark5' != bottleneck_name:
                        if add_to_list:         # 256 * 512
                            new_model_list[sanitized_name] = layer

                    if 'backbone.backbone.dark5.2.conv3' == full_name:
                        if add_to_list and 'backbone.backbone.dark5' != bottleneck_name: # (512 * 512)
                            new_model_list[sanitized_name] = layer

                    if 'backbone.lateral_conv0' == full_name:  # (512 * 256)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'backbone.C3_p4.conv3' == full_name and 'backbone.C3_p4' != bottleneck_name:  # (256 * 256)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'backbone.reduce_conv1' == full_name and 'backbone.C3_p3' != bottleneck_name:  # (256 * 128)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'backbone.C3_p3.conv3' == full_name:  # (128 * 128)
                        if add_to_list and 'backbone.C3_p3' != bottleneck_name:
                            new_model_list[sanitized_name] = layer
                            add_CP = False

                    if 'head.cls_convs.0.0' == full_name and 'head.cls_convs' in self.head  and bottleneck_name != 'backbone.C3_n4'  and bottleneck_name != 'backbone.C3_n3':  # (128 * 128)
                        if add_to_list:
                            if 'cls_convs' in bottleneck_name:
                                new_model_list[sanitized_name] = layer
                                break
                            elif 'stems_0' not in new_model_list:  # (128 * 128)
                                stems_0_layer = \
                                [l for n, l in model.named_modules() if 'stems.0' in n][0]
                                stems_0_name = 'stems_0'
                                new_model_list[stems_0_name] = stems_0_layer

                            new_model_list[sanitized_name] = layer

                    if 'head.cls_convs.1.0' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n3':  # (128 * 128)
                        if add_to_list:
                            if 'cls_convs' in bottleneck_name:
                                new_model_list[sanitized_name] = layer
                                break
                            elif 'stems_1' not in new_model_list:  # (256 * 128)
                                stems_1_layer = \
                                [l for n, l in model.named_modules() if 'stems.1' in n][0]
                                stems_1_name = 'stems_1'
                                new_model_list[stems_1_name] = stems_1_layer

                            new_model_list[sanitized_name] = layer


                    if 'head.cls_convs.2.0' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n4':  # (128 * 128)
                        if add_to_list:
                            if 'cls_convs' in bottleneck_name:
                                new_model_list[sanitized_name] = layer
                                break
                            elif 'stems_2' not in new_model_list:  # (512 * 128)
                                stems_2_layer = \
                                [l for n, l in model.named_modules() if 'stems.2' in n][0]
                                stems_2_name = 'stems_2'
                                new_model_list[stems_2_name] = stems_2_layer

                            new_model_list[sanitized_name] = layer


                    if 'head.cls_convs.0.1' == full_name and 'head.cls_convs' in self.head and bottleneck_name != 'backbone.C3_n3' and bottleneck_name != 'backbone.C3_n4':  # (128 * 128)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.cls_convs.1.1' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n3':  # (128 * 128)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.cls_convs.2.1' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n4':  # (128 * 128)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.cls_preds.0' == full_name and 'head.cls_convs' in self.head and bottleneck_name != 'backbone.C3_n3' and  bottleneck_name != 'backbone.C3_n4':       # (128 * 80)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.cls_preds.1' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n3':       # (128 * 80)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.cls_preds.2' == full_name and 'head.cls_convs' in self.head and bottleneck_name == 'backbone.C3_n4':       # (128 * 80)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if 'head.reg_convs.2.0' == full_name and 'head.reg_convs' in self.head and bottleneck_name == 'backbone.C3_n4':  # (128 * 128)
                        if add_to_list:
                            if 'reg_convs' in bottleneck_name:
                                new_model_list[sanitized_name] = layer
                                break
                            elif 'stems_2' not in new_model_list:  # (512 * 128)
                                stems_2_layer = \
                                [l for n, l in model.named_modules() if 'stems.2' in n][0]
                                stems_2_name = 'stems_2'
                                new_model_list[stems_2_name] = stems_2_layer

                            new_model_list[sanitized_name] = layer


                    if 'head.reg_convs.2.1' == full_name and 'head.reg_convs' in self.head and bottleneck_name == 'backbone.C3_n4':  # (128 * 128)
                        if add_to_list:
                            new_model_list[sanitized_name] = layer


                    if 'head.reg_preds.2' == full_name and 'head.reg_convs' in self.head and bottleneck_name == 'backbone.C3_n4':
                        if add_to_list:
                            new_model_list[sanitized_name] = layer

                    if full_name == bottleneck_name:
                        add_to_list = True

                    if isinstance(layer, nn.Module):
                        _collect(layer, full_name)

            _collect(model)
            return new_model_list

        new_model_list = traverse_and_collect(self.model, bottleneck)
        temp_model = nn.Sequential(new_model_list)


        for name, module in temp_model.named_modules():
            module.register_forward_hook(self._debug_hook(name))

        #return temp_model
        return temp_model

    def _get_gradients(self, acts, y, bottleneck_name, example=None):
        inputs = torch.autograd.Variable(torch.tensor(acts), requires_grad=True)
        cutted_model = self._get_cutted_model(bottleneck_name)
        cutted_model.eval()

        outputs = cutted_model(inputs.permute(0, 3, 1, 2))

        # for class-wise classification
        class_output = outputs[:, y[0], :, :]
        flatten_output = class_output.flatten(start_dim=1)

        # for bbox
        #outputs = outputs[:, 0:2, :, :]
        #flatten_output = outputs.flatten(start_dim=1)


        grads = -torch.autograd.grad(flatten_output.sum(), inputs)[0]

        #outputs_pooled = F.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)
        #outputs_target = outputs_pooled[:, y[0]]
        #grads = -torch.autograd.grad(outputs[:, y[0]].sum(), inputs)[0]

        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()

        return grads

    def _tcav_score(self, bn, concept, rnd, gradients):
        """Calculates and returns the TCAV score of a concept.

    Args:
      bn: bottleneck layer
      concept: concept name
      rnd: random counterpart
      gradients: Dict of gradients of tcav_score_images

    Returns:
      TCAV score of the concept with respect to the given random counterpart
    """
        vector = self.load_cav_direction(concept, rnd, bn)

        prod = np.sum(gradients[bn] * vector, -1)

        tcav_score = np.mean(prod < 0)

        return tcav_score

    def tcavs(self, test=True, sort=True, tcav_score_images=None):
        """Calculates TCAV scores for all discovered concepts and sorts concepts.

    This method calculates TCAV scores of all the discovered concepts for
    the target class using all the calculated cavs. It later sorts concepts
    based on their TCAV scores.

    Args:
      test: If true, perform statistical testing and removes concepts that don't
        pass
      sort: If true, it will sort concepts in each bottleneck layers based on
        average TCAV score of the concept.
      tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.

    Returns:
      A dictionary of the form {'bottleneck layer':{'concept name':
      [list of tcav scores], ...}, ...} containing TCAV scores.
    """
        tcav_scores = {bn: {} for bn in self.bottlenecks}

        randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_exp)]
        if tcav_score_images is None:  # Load target class images if not given
            raw_imgs = self.load_concept_imgs(self.target_class, 2 * self.max_imgs)
            tcav_score_images = raw_imgs[-self.max_imgs:]

        gradients = self._return_gradients(tcav_score_images)
        for bn in self.bottlenecks:

            for concept in self.dic[bn]['concepts'] + [self.random_concept]:

                def t_func(rnd):
                    score = self._tcav_score(bn, concept, rnd, gradients)
                    return score

                if self.num_workers:
                    pool = multiprocessing.Pool(self.num_workers)
                    tcav_scores[bn][concept] = pool.map(lambda rnd: t_func(rnd), randoms)
                else:
                    tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]
        if test:
            self.test_and_remove_concepts(tcav_scores)
        if sort:
            self._sort_concepts(tcav_scores)
        return tcav_scores

    def do_statistical_testings(self, i_ups_concept, i_ups_random):
        """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
        min_len = min(len(i_ups_concept), len(i_ups_random))

        _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
        return p

    def test_and_remove_concepts(self, tcav_scores):
        """Performs statistical testing for all discovered concepts.

    Using TCAV socres of the discovered concepts versurs the random_counterpart
    concept, performs statistical testing and removes concepts that do not pass

    Args:
      tcav_scores: Calculated dicationary of tcav scores of all concepts
    """
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts']:

                pvalue = self.do_statistical_testings \
                    (tcav_scores[bn][concept], tcav_scores[bn][self.random_concept])
                if pvalue > 0.01:
                    concepts_to_delete.append((bn, concept))
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)

    def delete_concept(self, bn, concept):
        """Removes a discovered concepts if it's not already removed.

    Args:
      bn: Bottleneck layer where the concepts is discovered.
      concept: concept name
    """
        self.dic[bn].pop(concept, None)
        if concept in self.dic[bn]['concepts']:
            self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

    def _concept_profile(self, bn, activations, concept, randoms):
        """Transforms data points from activations space to concept space.

    Calculates concept profile of data points in the desired bottleneck
    layer's activation space for one of the concepts

    Args:
      bn: Bottleneck layer
      activations: activations of the data points in the bottleneck layer
      concept: concept name
      randoms: random concepts

    Returns:
      The projection of activations of all images on all CAV directions of
        the given concept
    """

        def t_func(rnd):
            products = self.load_cav_direction(concept, rnd, bn) * activations
            return np.sum(products, -1)

        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            profiles = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
            profiles = [t_func(rnd) for rnd in randoms]
        return np.stack(profiles, axis=-1)

    def find_profile(self, bn, images, mean=True):
        """Transforms images from pixel space to concept space.

    Args:
      bn: Bottleneck layer
      images: Data points to be transformed
      mean: If true, the profile of each concept would be the average inner
        product of all that concepts' CAV vectors rather than the stacked up
        version.

    Returns:
      The concept profile of input images in the bn layer.
    """
        profile = np.zeros((len(images), len(self.dic[bn]['concepts']),
                            self.num_random_exp))
        class_acts = self.get_acts_from_images(
            images, bn).reshape([len(images), -1])
        randoms = ['random500_{}'.format(i) for i in range(self.num_random_exp)]
        for i, concept in enumerate(self.dic[bn]['concepts']):
            profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
        if mean:
            profile = np.mean(profile, -1)
        return profile

    def get_acts_from_images(self, imgs, bottleneck):
        """Run images in the model to get the activations.
    Args:
      imgs: a list of images
      model: a model instance
      bottleneck_name: bottleneck name to get the activation from
    Returns:
      numpy array of activations.
    """

        features_blobs.clear()

        img_batch_tensor = torchvision.transforms.functional.normalize(
            torch.tensor(imgs).permute(0, 3, 1, 2),
            mean=self.mean, std=self.std).float()

        _ = self.model(img_batch_tensor)

        #if bottleneck == 'backbone.backbone.stem':
        #    out = features_blobs[0].transpose(0, 2, 3, 1)
        #elif bottleneck == 'backbone.backbone.dark2.0':
        #    out = features_blobs[1].transpose(0, 2, 3, 1)

        #elif bottleneck == 'backbone.backbone.dark2':
        #    out = features_blobs[2].transpose(0, 2, 3, 1)
        #elif bottleneck == 'backbone.backbone.dark3':
        #    out = features_blobs[3].transpose(0, 2, 3, 1)
        #elif bottleneck == 'backbone.backbone.dark4':
        #    out = features_blobs[4].transpose(0, 2, 3, 1)
        #elif bottleneck == 'backbone.backbone.dark5.1':
        #    out = features_blobs[5].transpose(0, 2, 3, 1)

        #elif bottleneck == 'backbone.backbone.dark5':
        #    out = features_blobs[6].transpose(0, 2, 3, 1)

        # if bottleneck == 'backbone.C3_p4':
        #    out = features_blobs[0].transpose(0, 2, 3, 1)

        elif bottleneck == 'backbone.C3_p3':
            out = features_blobs[0].transpose(0, 2, 3, 1)

        elif bottleneck == 'backbone.C3_n3':
            out = features_blobs[1].transpose(0, 2, 3, 1)

        elif bottleneck == 'backbone.C3_n4':
            out = features_blobs[2].transpose(0, 2, 3, 1)

        else:
            out = features_blobs[0].transpose(0, 2, 3, 1)


        features_blobs.pop()
        return out


def get_layer_by_name(model, layer_name):
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        else:
            return None
    return layer


features_blobs = []


def hook_feature(module, input, output):
    # global features_blobs
    # print(f"Layer: {module.__class__.__name__}, Input shape: {input[0].shape}, Output shape: {output.shape}")
    features_blobs.append(output.data.cpu().numpy())
    # print(f"Length of features_blobs {len(features_blobs)}")


def get_yolox_model(model_name):
    if model_name == "yolox-s":
        depth = 0.33
        width = 0.50
    elif model_name == "yolox-m":
        depth = 0.67
        width = 0.75
    elif model_name == "yolox-l":
        depth = 1.0
        width = 1.0
    elif model_name == "yolox-x":
        depth = 1.33
        width = 1.25
    else:
        raise ValueError("Unsupported model type")

    # Initialize the model
    model = YOLOX(
        backbone=YOLOPAFPN(depth=depth, width=width),
        head=YOLOXHead(num_classes=80, width=width)
    )
    return model


def make_model(settings):
    model = None
    if len(settings.model_path) < 1:
        model = get_yolox_model(settings.model_to_run)
        print(f"YOLOX model {settings.model_to_run} loaded with default weights")
    else:
        print('Loading target model checkpoint from: {}'.format(settings.model_path))
        checkpoint = torch.load(settings.model_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            model = get_yolox_model(settings.model_to_run)
            if any('module' in key for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Model state dict loaded successfully")
        else:
            raise RuntimeError("Checkpoint format not recognized")

    # Ensuring model instance
    if model is None:
        raise ValueError("Failed to load the model. Check the checkpoint format.")

    # Move the model to CPU
    model.to(torch.device('cpu'))

    # Register hooks for specific layers in YOLOX backbone
    for name in settings.feature_names:
        new_name = ""
        if name == "head.cls_convs":
            new_name = name.replace("head.cls_convs", "head.cls_convs.2")
            layer = get_layer_by_name(model, new_name)
            if layer:
                print(f"Registering hook on layer: {new_name}, layer details: {layer}")
                layer.register_forward_hook(hook_feature)
            else:
                print(f"Layer {name} not found in model")
        layer = get_layer_by_name(model, name)
        if layer:
            print(f"Registering hook on layer: {name}, layer details: {layer}")
            layer.register_forward_hook(hook_feature)
        else:
            print(f"Layer {name} not found in model")

    model.eval()

    # for txt (COCO)
    with open(settings.labels_path, "r") as f:
        labels = f.read().splitlines()
        model.labels = labels
        model.label_to_id = {v: k for (k, v) in enumerate(model.labels)}

    # for csv (GTSRB)
    #meta_df = pd.read_csv(settings.labels_path)
    #class_ids = meta_df['ClassId'].unique()
    #label_list = [str(class_id) for class_id in sorted(class_ids)]
    #model.labels = label_list
    #model.label_to_id = {v: k for (k, v) in enumerate(model.labels)}

    # Print statements to check labels and label_to_id mapping
    print(f"Labels assigned to model: {model.labels}")
    print(f"Label to ID mapping: {model.label_to_id}")

    return model
