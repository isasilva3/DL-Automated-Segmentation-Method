img_dir="//home//imoreira//JupyterExercise"
out_dir="//home//imoreira//JupyterExercise"

def arg_parser():
    img_dir = "//home//imoreira//JupyterExercise"
    out_dir = "//home//imoreira//JupyterExercise"
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='//home//imoreira//JupyterExercise')
    parser.add_argument('out_dir', type=str,
                        help='//home//imoreira//JupyterExercise')
    parser.add_argument('-a', '--axis', type=int, default=2,
                        help='axis on which to stack the 2d images')
    return parser


def split_filename(filepath):
    img_dir = "//home//imoreira//JupyterExercise"
    out_dir = "//home//imoreira//JupyterExercise"
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main():
    try:
        args = arg_parser().parse_args()
        img_dir = pathlib.Path(args.img_dir)
        fns = sorted([str(fn) for fn in img_dir.glob('*.tif*')])
        if not fns:
            raise ValueError(f'img_dir ({args.img_dir}) does not contain any .tif or .tiff images.')
        imgs = []
        for fn in fns:
            _, base, ext = split_filename(fn)
            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            if img.ndim != 2:
                raise Exception(f'Only 2D data supported. File {base}{ext} has dimension {img.ndim}.')
            imgs.append(img)
        img = np.stack(imgs, axis=args.axis)
        nib.Nifti1Image(img,None).to_filename(os.path.join(args.out_dir, f'{base}.nii.gz'))
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

