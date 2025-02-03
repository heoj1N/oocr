import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import albumentations as A
import shutil

# -------------
# BENCHMARK DATASETS
# -------------

class IAMDataset(Dataset):
    """
    IAM words/sentences dataset
    https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database 
    ---------------------
    dataroot
    | words
    |  | a01
    |  |  | a01-000u
    |  |  |  | a01-000u-00-00.png
    |  |  |  | a01-000u-00-01.png
    |  |  |  | a01-000u-00-02.png
    | words.txt
    ---------------------
    """
    def __init__(self, df: pd.DataFrame, processor, max_target_length=128, split='train', use_augmentation=False):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'file_path' and 'transcription'
            processor: TrOCRProcessor
            max_target_length (int): Maximum length of tokenized text
            split (str): 'train', 'eval', or 'test'
            use_augmentation (bool): Whether to use data augmentation
        """
        self.processor = processor
        self.max_target_length = max_target_length
        self.split = split
        self.transform = get_train_augmentations() if split == 'train' and use_augmentation else None
        
        # Filter out any invalid or too-small images
        # i.e. 1x1 pixel images exist in IAM
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = row['file_path']
            try:
                # Attempt to open and convert to RGB
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img)
                    
                    # Check for non-trivial dimensions
                    if arr.shape[0] > 1 and arr.shape[1] > 1:
                        valid_rows.append(row)
                    else:
                        print(f"[WARNING] Skipped tiny image '{image_path}' with shape {arr.shape}")
            except Exception as e:
                print(f"[ERROR] Could not open or convert '{image_path}'. Skipping. ({e})")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[INFO] Kept {len(self.df)}/{len(df)} images after filtering.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['file_path']
        text = str(self.df.iloc[idx]['transcription'])

        image = Image.open(image_path).convert("RGB")
        
        # Apply augmentations if enabled
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = Image.fromarray(augmented['image'])

        encoding = self.processor(
            image,
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(dim=0)
            
        return encoding
    
    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create dataset from IAM format"""             
        labels_file = os.path.join(datapath, "words.txt")
        try:
            df = pd.read_csv(labels_file, sep='\s+', header=None, comment='#',
                names=["word_id", "segmentation_status", "graylevel", "x", "y", 
                    "width", "height", "grammatical_tag", "transcription"],
                on_bad_lines='skip', engine='python')
            logger.info(f"Dataset loaded successfully with {len(df)} entries.")
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            raise

        df['file_path'] = df['word_id'].apply(
            lambda x: os.path.join(datapath, "words",
                x.split('-')[0],                # "a01"
                x.split('-')[0] + "-" + x.split('-')[1],  # "a01-000u"
                x + ".png"                      # "a01-000u-00-00.png"
            )
        )
        df = validate_files(df, logger)
        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        
        datasets = {
            'train': IAMDataset(df=train_df, processor=processor, max_target_length=128, split='train', use_augmentation=args.use_augmentation),
            'eval': IAMDataset(df=eval_df, processor=processor, max_target_length=128, split='eval', use_augmentation=args.use_augmentation),
            'test': IAMDataset(df=test_df, processor=processor, max_target_length=128, split='test', use_augmentation=args.use_augmentation)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', header=None, 
                names=["file_name", "text"]
            )
            new_test_df['file_name'] = new_test_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            new_test_df['file_path'] = new_test_df['file_name'].apply(
                lambda x: os.path.join(args.testdata, "images", x)
            )
            # Rename 'text' column to match IAM dataset's 'transcription' column
            new_test_df = new_test_df.rename(columns={'text': 'transcription'})
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = IAMDataset(df=new_test_df, processor=processor, max_target_length=128)
        
        return datasets

class MNISTDataset(Dataset):
    """
    MNIST Dataset
    https://pytorch.org/vision/0.20/generated/torchvision.datasets.MNIST.html 
    """
    def __init__(self, root_dir, processor, df=None, max_target_length=128, downloadable=True):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = df
        self.downloadable = downloadable
        
        # Define transforms to convert to RGB (MNIST is grayscale)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.ToPILImage()  # Convert back to PIL for processor
        ])
        
        self.mnist = MNIST(root_dir, train=True, download=True)

    def __len__(self):
        return len(self.df) if self.df is not None else len(self.mnist)

    def __getitem__(self, idx):
        if self.df is not None:
            image_id = self.df.iloc[idx]['image_id']
            label = self.df.iloc[idx]['transcription']
            image, _ = self.mnist[image_id]
        else:
            image, label = self.mnist[idx]
            label = str(label)
        
        # Convert to RGB and preprocess
        image = self.transform(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize label
        labels = self.processor.tokenizer(
            text_target=label,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace padding token id with -100 for loss calculation
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create MNIST dataset splits"""
        
        # Load the full dataset
        mnist = MNIST(datapath, train=True, download=True)
        
        # Create a DataFrame similar to other datasets
        df = pd.DataFrame({
            'image_id': range(len(mnist)),
            'transcription': [str(label) for _, label in mnist],
        })
        
        # Split dataset
        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        
        datasets = {
            'train': MNISTDataset(datapath, processor, df=train_df, use_augmentation=args.use_augmentation),
            'eval': MNISTDataset(datapath, processor, df=eval_df, use_augmentation=args.use_augmentation),
            'test': MNISTDataset(datapath, processor, df=test_df, use_augmentation=args.use_augmentation)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', 
                header=None, 
                names=["file_name", "text"]
            )
            new_test_df = new_test_df.rename(columns={'text': 'transcription'})
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = MNISTDataset(args.testdata, processor, df=new_test_df)
        
        return datasets

class SROIEDataset(Dataset):
    """
    SROIE (Scanned Receipt OCR and Information Extraction) Dataset
    https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data
    Data: ICDAR 2019 Competition Dataset
    ---------------------
    dataroot
    ├── train
    │   ├── box/
    │   │   └── *.txt (bounding box coordinates and text)
    │   └── img/
    │       └── *.jpg (receipt images)
    └── test
        ├── box/
        │   └── *.txt
        └── img/
            └── *.jpg
    ---------------------
    """
    def __init__(self, root_dir, processor, max_target_length=128, df=None, 
                 split='train', use_augmentation=False):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.split = split
        self.transform = get_train_augmentations() if split == 'train' and use_augmentation else None
        
        if df is not None:
            self.df = df
        else:
            self.df = self._load_dataset()

    def __len__(self):
        return len(self.df)

    def _load_dataset(self):
        """Load and process the SROIE dataset"""
        split_dir = os.path.join(self.root_dir, self.split)
        img_dir = os.path.join(split_dir, 'img')
        box_dir = os.path.join(split_dir, 'box')
        
        data = []
        
        # Process each image and its corresponding box file
        for img_file in os.listdir(img_dir):
            if not img_file.endswith('.jpg'):
                continue
                
            base_name = os.path.splitext(img_file)[0]
            box_file = os.path.join(box_dir, f"{base_name}.txt")
            
            if not os.path.exists(box_file):
                continue
                
            # Read bounding boxes and text
            try:
                # Try UTF-8 first
                with open(box_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                try:
                    # Try latin-1 if UTF-8 fails
                    with open(box_file, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    # If both fail, try with errors='ignore'
                    with open(box_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
            
            # Process the lines
            for line in lines:
                parts = line.strip().split(',', 8)
                if len(parts) != 9:
                    continue
                        
                coords = [float(x) for x in parts[:8]]
                text = parts[8]
                
                data.append({
                    'image_path': os.path.join(img_dir, img_file),
                    'coords': coords,
                    'text': text,
                    'base_name': base_name
                })
                
        return pd.DataFrame(data)

    def _crop_image(self, image, coords):
        """Crop image using quadrilateral coordinates"""
        # Convert coordinates to numpy array
        coords = np.array(coords).reshape(-1, 2)
        
        # Get bounding box
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        # Crop with some padding
        padding = 2
        x_min, y_min = max(0, int(x_min - padding)), max(0, int(y_min - padding))
        x_max, y_max = min(image.width, int(x_max + padding)), min(image.height, int(y_max + padding))
        
        return image.crop((x_min, y_min, x_max, y_max))

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        file_path = self.df['file_path'].iloc[idx]
        text = self.df['transcription'].iloc[idx]
        coords = self.df['coords'].iloc[idx]  # Get coordinates for cropping

        # Load and convert image
        image = Image.open(file_path).convert("RGB")
        
        # Crop the image using the coordinates
        cropped_image = self._crop_image(image, coords)
        
        # Apply augmentations if enabled
        if self.transform:
            image_np = np.array(cropped_image)
            augmented = self.transform(image=image_np)
            cropped_image = Image.fromarray(augmented['image'])

        # Process image and text
        encoding = self.processor(
            cropped_image,  # Use the cropped image
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(dim=0)
            
        return encoding

    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create dataset from SROIE format"""
        try:
            # Check if dataset needs to be downloaded
            if not os.path.exists(datapath) or not os.listdir(datapath):
                logger.info("SROIE dataset not found. Attempting to download...")
                try:
                    import kaggle
                    os.makedirs(datapath, exist_ok=True)
                    
                    # Download dataset using kaggle API
                    kaggle.api.authenticate()
                    kaggle.api.dataset_download_files(
                        'urbikn/sroie-datasetv2',
                        path=datapath,
                        unzip=True
                    )
                    
                    nested_dir = os.path.join(datapath, 'SROIE2019')  # The nested directory created
                    if os.path.exists(nested_dir):
                        try:
                            # Move contents up one level using shutil
                            for item in os.listdir(nested_dir):
                                src = os.path.join(nested_dir, item)
                                dst = os.path.join(datapath, item)
                                if os.path.exists(dst):  # If destination already exists, remove it
                                    if os.path.isdir(dst):
                                        shutil.rmtree(dst)
                                    else:
                                        os.remove(dst)
                                shutil.move(src, dst)  # Use shutil.move instead of os.rename
                            # Remove now-empty nested directory
                            shutil.rmtree(nested_dir)
                            logger.info("Fixed nested directory structure")
                        except Exception as e:
                            logger.warning(f"Could not fix directory structure: {e}")
                            logger.info("You may need to manually move files from SROIE2019/SROIE2019/* to SROIE2019/*")
                    
                    if not os.path.exists(os.path.join(datapath, 'train')) or \
                       not os.path.exists(os.path.join(datapath, 'test')):
                        raise RuntimeError("Dataset download seems incomplete")
                    logger.info(f"Successfully downloaded SROIE dataset to: {datapath}")
                    
                except ImportError:
                    logger.error("kaggle package not installed. Install with: pip install kaggle")
                    raise
                except Exception as e:
                    logger.error(f"Failed to download SROIE dataset: {e}")
                    logger.info("To use kaggle API, you need to:")
                    logger.info("1. Install kaggle: pip install kaggle")
                    logger.info("2. Set up Kaggle credentials:")
                    logger.info("   - Go to https://www.kaggle.com/settings/account")
                    logger.info("   - Click 'Create New Token' to download kaggle.json")
                    logger.info("   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)")
                    logger.info("   - Ensure file permissions are secure: chmod 600 ~/.kaggle/kaggle.json")
                    raise

            # Process training data
            train_data = []
            train_dir = os.path.join(datapath, 'train')
            train_img_dir = os.path.join(train_dir, 'img')
            train_box_dir = os.path.join(train_dir, 'box')

            # Process each image in training set
            for img_file in os.listdir(train_img_dir):
                if not img_file.endswith('.jpg'):
                    continue
                
                base_name = os.path.splitext(img_file)[0]
                box_file = os.path.join(train_box_dir, f"{base_name}.txt")
                
                if not os.path.exists(box_file):
                    logger.warning(f"Missing box file for {img_file}")
                    continue
                
                # Read bounding boxes and text with multiple encoding attempts
                try:
                    # Try UTF-8 first
                    with open(box_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    logger.error(f"Failed to read {box_file} with UTF-8 encoding")
                    continue
                
                # Process the lines
                for line in lines:
                    parts = line.strip().split(',', 8)
                    if len(parts) != 9:
                        continue
                        
                    coords = [float(x) for x in parts[:8]]
                    text = parts[8]
                    
                    train_data.append({
                        'file_path': os.path.join(train_img_dir, img_file),
                        'coords': coords,
                        'transcription': text,
                        'base_name': base_name
                    })

            # Convert to DataFrame and apply fraction
            train_df = pd.DataFrame(train_data)
            train_df = train_df[['file_path', 'transcription', 'coords']]
            train_df = train_df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
            
            # Split training data into train and eval
            train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=args.seed)
            
            # Create datasets
            datasets = {
                'train': cls(datapath, processor, df=train_df, split='train', use_augmentation=args.use_augmentation),
                'eval': cls(datapath, processor, df=eval_df, split='train', use_augmentation=args.use_augmentation),
                'test': None  # Will be filled below
            }
            
            # Process test data or use provided test data
            if args.testdata:
                logger.info(f"Using provided test data from {args.testdata}")
                new_test_df = pd.read_csv(
                    os.path.join(args.testdata, "labels.txt"), 
                    sep='\t', header=None, 
                    names=["file_name", "text"]
                )
                new_test_df['file_name'] = new_test_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
                new_test_df['file_path'] = new_test_df['file_name'].apply(
                    lambda x: os.path.join(args.testdata, "images", x)
                )
                new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
                datasets['test'] = cls(args.testdata, processor, df=new_test_df)
            else:
                # Use SROIE test set
                test_data = []
                test_dir = os.path.join(datapath, 'test')
                test_img_dir = os.path.join(test_dir, 'img')
                test_box_dir = os.path.join(test_dir, 'box')
                
                # Process each image in test set
                for img_file in os.listdir(test_img_dir):
                    if not img_file.endswith('.jpg'):
                        continue
                    
                    base_name = os.path.splitext(img_file)[0]
                    box_file = os.path.join(test_box_dir, f"{base_name}.txt")
                    
                    if not os.path.exists(box_file):
                        logger.warning(f"Missing box file for {img_file}")
                        continue
                    
                    # Read bounding boxes and text
                    try:
                        # Try UTF-8 first
                        with open(box_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                    except UnicodeDecodeError:
                        logger.error(f"Failed to read {box_file} with UTF-8 encoding")
                        continue
                    
                    # Process the lines
                    for line in lines:
                        parts = line.strip().split(',', 8)
                        if len(parts) != 9:
                            continue
                                
                        coords = [float(x) for x in parts[:8]]
                        text = parts[8]
                        
                        test_data.append({
                            'file_path': os.path.join(test_img_dir, img_file),
                            'coords': coords,
                            'transcription': text,
                            'base_name': base_name
                        })
                
                test_df = pd.DataFrame(test_data)
                test_df = test_df[['file_path', 'transcription', 'coords']]
                test_df = test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
                datasets['test'] = cls(datapath, processor, df=test_df, split='test', use_augmentation=args.use_augmentation)
            
            # Log dataset sizes
            logger.info(f"Number of training examples: {len(datasets['train'])}")
            logger.info(f"Number of validation examples: {len(datasets['eval'])}")
            logger.info(f"Number of test examples: {len(datasets['test'])}")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error creating SROIE dataset: {e}")
            raise

# -------------
# CUSTOM DATASETS
# -------------

class SimpleDataset(Dataset):
    """
    Synthetic dataset generated with generator.py script
    Data (romanian language): 
    https://drive.google.com/drive/folders/1ErvjszLBqVIrO7wnsVUc6zWv5CtPmgF_?usp=drive_link
    ---------------------
    dataroot
    | images
    |  | image0.png
    |  | image1.png
    | labels.txt
    ---------------------
    labels.txt contains:
        'image0.png'\t'<text label>'
        'image1.png'\t'<text label>'
        ...
    """
    def __init__(self, root_dir, processor, labels_file="labels.txt", 
                 max_target_length=128, df=None, split='train', 
                 use_augmentation=False):
        """
        Initialize the OCR dataset with preprocessing.

        Args:
            root_dir (str): Root directory containing the dataset.
            processor: A processor for image and text (e.g., TrOCRProcessor).
            labels_file (str): Name of the labels file in the dataset.
            max_target_length (int): Maximum length of tokenized text.
            df (pd.DataFrame, optional): If provided, use this dataframe directly 
                                         instead of reading and splitting.
            split (str): 'train', 'eval', or 'test'
            use_augmentation (bool): Whether to use data augmentation
        """
        self.root_dir = root_dir
        self.processor = processor
        self.labels_file = os.path.join(root_dir, labels_file)
        self.max_target_length = max_target_length
        self.split = split
        self.transform = get_train_augmentations() if split == 'train' and use_augmentation else None

        os.makedirs(self.root_dir, exist_ok=True)

        if df is not None:
            # Use the provided DataFrame directly
            self.df = df
        else:
            # Ensure labels file exists
            if not self.labels_file.exists():
                raise FileNotFoundError(f"Labels file '{self.labels_file}' not found.")
            # If no dataframe is provided, load from file
            df = pd.read_csv(self.labels_file, sep='\t', header=None, names=["file_name", "text"])
            df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            df['file_path'] = df['file_name'].apply(lambda x: os.path.join(self.root_dir, "images", x))
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = Path(self.df['file_path'].iloc[idx])
        text = self.df['text'].iloc[idx]
        if not file_path.exists():
            raise FileNotFoundError(f"Image file '{file_path}' not found.")

        image = Image.open(file_path).convert("RGB")
        
        # Apply augmentations if enabled
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = Image.fromarray(augmented['image'])

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            text_target=text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
    
    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create dataset from custom format with labels.txt file"""
        train_file = os.path.join(datapath, "labels.txt")
        df = pd.read_csv(train_file, sep='\t', header=None, names=["file_name", "text"])
        df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
        df['file_path'] = df['file_name'].apply(lambda x: os.path.join(datapath, "images", x))

        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        datasets = {
            'train': SimpleDataset(root_dir=datapath, processor=processor, df=train_df, split='train', use_augmentation=args.use_augmentation),
            'eval': SimpleDataset(root_dir=datapath, processor=processor, df=eval_df, split='eval', use_augmentation=args.use_augmentation),
            'test': SimpleDataset(root_dir=datapath, processor=processor, df=test_df, split='test', use_augmentation=args.use_augmentation)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', header=None, 
                names=["file_name", "text"]
            )
            new_test_df['file_name'] = new_test_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            new_test_df['file_path'] = new_test_df['file_name'].apply(
                lambda x: os.path.join(args.testdata, "images", x)
            )
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = SimpleDataset(root_dir=args.testdata, processor=processor, df=new_test_df)
        
        return datasets
    
    @staticmethod
    def _fix_file_extension(file_name):
        """
        Fix common file extension issues.
        """
        if file_name.endswith('jp'):  # Handle cases like `.jp` files
            return file_name + 'g'
        return file_name

# -------------
# UTILS
# -------------

def validate_files(df, logger):
    """Validate that all image files exist"""
    missing_files = df[~df['file_path'].apply(os.path.exists)]
    if not missing_files.empty:
        logger.warning(f"Missing {len(missing_files)} files. Example: {missing_files.iloc[0]['file_path']}")
    df = df[df['file_path'].apply(os.path.exists)]
    if df.empty:
        logger.error("No valid image files found in dataset.")
        raise ValueError("Empty dataset after validation")
    return df

def init_dataset(args, processor, fraction, test_frac, logger):
    """Factory function to create appropriate dataset"""
    dataset_classes = {
        'custom': SimpleDataset,
        'IAM': IAMDataset,
        'MNIST': MNISTDataset,
        'SROIE': SROIEDataset
    }
    if args.dataset not in dataset_classes:
        raise ValueError(f"Dataset {args.dataset} not available")
    
    dataset_class = dataset_classes[args.dataset]
    datasets = dataset_class.create_dataset(args.data, processor, fraction, test_frac, args, logger)
    logger.info(f"Number of training examples: {len(datasets['train'])}")
    logger.info(f"Number of validation examples: {len(datasets['eval'])}")
    logger.info(f"Number of test examples: {len(datasets['test'])}")
    if args.use_augmentation:
        logger.info(f"Using augmentations")
    return datasets

def get_dataset(args):
    """Getter function for appropriate dataset"""
    dataset_classes = {
            'custom': SimpleDataset,
            'IAM': IAMDataset,
            'MNIST': MNISTDataset
        }
    if args.dataset not in dataset_classes:
        raise ValueError(f"Dataset {args.dataset} not available")
    
    dataset_class = dataset_classes[args.dataset]
    return dataset_class

def get_train_augmentations():
    """Define comprehensive training augmentations for OCR training"""
    return A.Compose([
        # Geometric Transformations (subtle to preserve text readability)
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=2, border_mode=0, value=(255, 255, 255), p=0.7),
            A.GridDistortion(num_steps=5, distort_limit=0.05, border_mode=0, value=(255, 255, 255), p=0.5),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, border_mode=0, value=(255, 255, 255), p=0.5),
            A.ElasticTransform(alpha=1, sigma=10, border_mode=0, value=(255, 255, 255), p=0.3),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=0, pad_val=(255, 255, 255), p=0.3),
        ], p=0.6),

        # Color and Intensity Transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.7),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Equalize(p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ], p=0.7),

        # Noise and Texture
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), mean=0, per_channel=True, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.93, 1.07), per_channel=True, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
            A.Posterize(num_bits=4, p=0.3),
        ], p=0.6),

        # Quality and Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 3), p=0.6),
            A.MotionBlur(blur_limit=3, p=0.4),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.AdvancedBlur(blur_limit=(3, 5), p=0.3),
            A.GlassBlur(sigma=0.5, max_delta=2, iterations=1, p=0.2),
        ], p=0.5),

        # Paper-like Effects and Degradation
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=2, max_width=2, min_holes=2, p=0.5),
            A.Spatter(mean=0.05, std=0.02, gauss_sigma=2, cutout_threshold=0.3, intensity=0.3, p=0.4),
            A.GridDropout(ratio=0.1, unit_size_min=2, unit_size_max=4, p=0.3),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
            A.ToGray(p=0.2),
        ], p=0.4),

        # Color Adjustments and Artifacts
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
            A.ChannelShuffle(p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            A.FancyPCA(alpha=0.1, p=0.2),
        ], p=0.4),

        # Random Shadows and Lighting
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.1, p=0.2),
        ], p=0.3),

        # Weather and Environmental Effects
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=8, drop_width=1, drop_color=(200, 200, 200), p=0.2),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2, brightness_coeff=1.5, p=0.2),
        ], p=0.2),

        # Ensure consistent size and padding
        A.PadIfNeeded(min_height=64, min_width=64, border_mode=0, value=(255, 255, 255), p=1.0),
    ])
