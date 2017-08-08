import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *

parser = argparse.ArgumentParser(description='Create pickles from images.')
parser.add_argument('--db_root',     metavar='db_root',     type=str,       help='Path to dataset root')
parser.add_argument('--pickle_file', metavar='pickle_file', type=str,       help='Pickle filename')
parser.add_argument('--is_invert',   dest='is_invert',      default='True', type=str, help='Invert image')

args = parser.parse_args()
db_root = args.db_root
pickle_filename = args.pickle_file
is_invert = args.is_invert

print('\nCreating pickles...')

image_folders = [
    os.path.join(db_root, d) for d in sorted(os.listdir(db_root))
    if os.path.isdir(os.path.join(db_root, d))]
dataset = maybe_pickle(image_folders, 1000, is_invert)

print('\nChecking frequencies of image sets...')

npickles = len(dataset)
stats = np.empty(shape=npickles, dtype=np.int64)

for i in np.arange(npickles):
    f = open(dataset[i], 'rb')
    letter_set = pickle.load(f)
    stats[i] = letter_set.shape[0]

print(stats)

print('\nMerging pickles and creating training and validation sets...')

train_size = 29376
valid_size = 7200

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(dataset, train_size, valid_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)

print('\nRandomizing training and validation sets...')

train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#print('\nDisplaying image samples...')
#fig = plt.figure()
#gs = gridspec.GridSpec(4, 16)

#for i in np.arange(len(dataset)):
#    """ Pick each pickle file, and load it into memory """
#    pickle_file = dataset[i]
#    f = open(pickle_file, 'rb')
#    letter_set = pickle.load(f)
#
#    """ Pick a random sample from the loaded set of images and display using matplotlib """
#    sample_idx = np.random.randint(len(letter_set))
#    sample_img = letter_set[sample_idx, :, :]
#
#    fig.add_subplot(gs[i])
#    plt.imshow(sample_img, cmap='gray')
#plt.show()

print('\nSaving merged dataset...')

pickle_file = os.path.join(os.path.dirname(db_root), pickle_filename)

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
