import os

def write_top(all_results, outputpath):

    out_dir = os.path.join(outputpath, 'top')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'townCentreOut.top')

    with open(out_file, 'w') as f_out:
        for im_res in all_results:
            im_name = im_res['imgname']
            for human in im_res['result']:
                person_id = human['person_id']
                bbox = human['bbox'].numpy().tolist()
                f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n'
                            % (person_id, im_name, 1, 1, bbox[0], bbox[1], bbox[2], bbox[3]))