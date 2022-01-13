import argparse
from imageprocessing import ImageProcessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-print_debug_info", action="store_true", help="print some debugging info")
    parser.add_argument("-save_debug_imgs" , action="store_true", help="save some debugging images")
    opt = parser.parse_args()    
    print_debug_info = opt.print_debug_info
    save_debug_imgs  = opt.save_debug_imgs
    
    img_processing = ImageProcessing(save_debug_imgs=False)
    #img_processing.run()
    #img_processing.plot_distribution()

if __name__ == '__main__':
    main()
    
    
    
    # no final: rever python meetup (PEP8)
    # remove comments
