import argparse
from imageprocessing import ImageProcessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_debug_imgs" , action="store_true", help="save some debugging images")
    opt = parser.parse_args()
    
    img_processing = ImageProcessing(save_debug_imgs=opt.save_debug_imgs)
    img_processing.run()
    img_processing.plot_distribution()

if __name__ == '__main__':
    main()
