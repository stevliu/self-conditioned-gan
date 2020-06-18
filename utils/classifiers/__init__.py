from classifiers import stacked_mnist, cifar, places, imagenet

classifier_dict = {
    'stacked_mnist': stacked_mnist.Classifier,
    'cifar': cifar.Classifier, 
    'places': places.Classifier,
    'imagenet': imagenet.Classifier
}