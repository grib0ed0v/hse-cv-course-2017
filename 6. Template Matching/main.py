import cv2
from utils import *
import anotherCluster as acl
import argparse


def get_global_cluster(matched_clusters, source_kp_clusters, source_dscr_clusters, cluster):
    """
    Get global cluster that include all key points from matched cluster

    :param matched_clusters:  clusters of matched key points
    :param source_kp_clusters: cluters of kye point
    :param source_dscr_clusters: clusters of descriptors
    :param cluster: numbers of cluster from matched_clusters
    :return: 
    """
    clusters = set()
    for kp in matched_clusters[cluster]:
        for i in range(source_kp_clusters.shape[0]):
            if kp in source_kp_clusters[i]:
                clusters.add(i)
    result_kp = []
    result_descr = []
    for cl in clusters:
        result_kp = np.append(result_kp, source_kp_clusters[cl])
        for d in source_dscr_clusters[cl]:
            result_descr.append(d)
    return np.array(result_kp), np.array(result_descr), clusters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_file','-t', default='pictures/C.png',help="template file")
    parser.add_argument('--target_file', '-f', default = 'pictures/C8.jpg',help= "target file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.template_file == None or args.target_file == None:
        raise IOError("Set please template and target files")

    # load images
    coke, query, C, gquery = load_etalon_query(args.template_file, args.target_file)
    # get pey points for template and source images
    kpc, desc, kpq, desq, src_pts, dst_pts, matches = keypoints_match(C, gquery)



    for ii in range(0, 10):

        matches = matching(desc, desq)
        # get list of points from source key points
        spoints_lst = get_points(kpq)
        # clustering source key points
        slabels, sncluster = acl.meanshift_clustering(spoints_lst, quantile=0.01, n_samples=500)
        if sncluster is None:
            break
        # split all key points and descriptors into clusters
        skp_clusters = clsplite(kpq, slabels, sncluster)
        sdscr_clusters = clsplite(desq, slabels, sncluster)



        # get matched key points
        mkpc, mkpq = get_mached_kpoints(kpc, kpq, matches)
        # get list of points from matched key points
        mspoints_lst = get_points(mkpq)
        # get descriptors of matched key points
        mdscr = np.array([desq[m.trainIdx] for m in matches])


        # clustering matched key points
        smlabels, smncluster = acl.meanshift_clustering(mspoints_lst, quantile=0.13, n_samples=500)
        if smncluster is None:
            break

        # split all key points and descriptors into clusters
        smkp_clusters = clsplite(mkpq, smlabels, smncluster)


        cluster = 0

        # get global cluster of key points and descriptors
        kp_glcluster, descr_glcluster, cls = get_global_cluster(smkp_clusters, skp_clusters, sdscr_clusters, cluster)
        # get matches from template to global cluster
        global_matches = matching(desc, descr_glcluster)
        # get list of matched key points
        template_matched_points, global_matched_points = get_mached_kpoints(kpc, kp_glcluster, global_matches)

        left, right = generate_template_shape(global_matched_points)

        cv2.rectangle(query, left, right, (0, 255, 0), thickness=5, lineType=8, shift=0)

        kpq, desq, slabels = remove_cluster(kpq, desq, slabels, cls)


    cv2.namedWindow('cluster', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cluster', 980, 720)
    cv2.imshow('cluster', query)
    cv2.waitKey()

if __name__ == '__main__':
    main()
