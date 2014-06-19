import sklearn
from sklearn.cluster import DBSCAN
import numpy as np
import sklearn.datasets as sd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Ward
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def main():
  trainDataFile = '/home/priya/ml/data/trainData'
  categories = ['Entrance', 'Physiotherapy', 'General', 'Reception', 'ChildrenArea', 'ParkingArea', 'Neurology', 'Gynecology', 'Pediatrics', 'Oncology', 'Food', 'Finance', 'Anesthetic', 'Hematology', 'Cardiology', 'Dermatology', 'Psychiatrist'
              ]
  department_train =  sd.load_files(trainDataFile, categories=categories)
  #print department_train.target_names
  #print len(department_train.data)
  #print len(department_train.filenames)
  #print("\n".join(department_train.data[0].split("\n")[:1]))
  #print(department_train.target_names[department_train.target[0]])
  #print department_train.target[:10]
  count_vect = CountVectorizer(ngram_range=(1,2)) #ngram_range=(1,2)  ngram_range=(1,3)
  X_train_counts = count_vect.fit_transform(department_train.data)
  #print X_train_counts.shape
  #print count_vect.vocabulary_.get(u'algorithm')
  labels = department_train.target
  true_k = np.unique(labels).shape[0]

  tf_transformer = TfidfTransformer(use_idf=False)
  X_train_tf = tf_transformer.transform(X_train_counts)
  #print X_train_tf.shape
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  #print X_train_tfidf.shape  
  #print 'array printing'
  #print X_train_tfidf.toarray();
  lsa = TruncatedSVD()
  X_train_lsa= lsa.fit_transform(X_train_counts)
  # Vectorizer results are normalized, which makes KMeans behave as
  # spherical k-means for better results. Since LSA/SVD results are
  # not normalized, we have to redo the normafrom sklearn.preprocessing import StandardScaler
  X_train_lsa = Normalizer(copy=False).fit_transform(X_train_lsa)

  #######classification#####################
  #clf = MultinomialNB().fit(X_train_tfidf, department_train.target)
  #testDataFile = ['Medicines are inadequate in the hospital', 'Poor facility at toilets that need to take care', 'BB is highly affected by electric faults', 'Anesthesia dept needs urgent cleaning', 'are for car parking have puddle', 'food is stinking at mess', 'Womens dr stethoscope is missing', 'brain dr cabin needs one stand maintenance', 'electric faults in skin dept']

  #X_new_counts = count_vect.transform(testDataFile)
  #X_new_tfidf = tfidf_transformer.transform(X_new_counts)
  #predicted = clf.predict(X_new_tfidf)
  #for doc, category in zip(testDataFile, predicted):
  #   print('%r => %s' % (doc, department_train.target_names[category]))
  #############################################

  ############### k-means clustering##################
  #km = KMeans( init='k-means++', max_iter=100, n_init=1
  #              )
  #print("Clustering sparse data with %s" % km)
  #km.fit(X_train_lsa)
  #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
  #print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
  #print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
  #print("Adjusted Rand-Index: %.3f"
  #  % metrics.adjusted_rand_score(labels, km.labels_))
  #print("Silhouette Coefficient: %0.3f"
  #  % metrics.silhouette_score(X_train_lsa, labels))
  ############################################

  ####### Affinity propagation clustering ######
  #af = AffinityPropagation(preference=-90).fit(X_train_lsa)
  #print af.shape
  #cluster_centers_indices = af.cluster_centers_indices_ # should be an array
  #n_clusters_ = len(cluster_centers_indices)
  #for x in range(0,n_clusters_):
  #  print "cluster's no %d" % (x)
  #lab = af.labels_
  #for x in range(0,n_clusters_):
  #  print "cluster's label %d" % lab[x]
  
  #print('Estimated number of clusters: %d' % n_clusters_)
  #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, lab))
  #print("Completeness: %0.3f" % metrics.completeness_score(labels, lab))
  #print("V-measure: %0.3f" % metrics.v_measure_score(labels, lab))
  #print("Adjusted Rand Index: %0.3f"
  #  % metrics.adjusted_rand_score(labels, lab))
  #print("Adjusted Mutual Information: %0.3f"
  #  % metrics.adjusted_mutual_info_score(labels, lab))
  #print("Silhouette Coefficient: %0.3f"
  #  % metrics.silhouette_score(X_train_tfidf, lab, metric='sqeuclidean'))
  ###########################################

  ############ Mean Shift clustering ##########
  bandwidth = estimate_bandwidth(X_train_tfidf, quantile=0.2)
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  ms.fit(X_train_tfidf)
  lab = ms.labels_
  cluster_centers = ms.cluster_centers_  
  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)
  print("number of estimated clusters : %d" % n_clusters_)
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, lab))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, lab))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, lab))
  print("Adjusted Rand Index: %0.3f"
    % metrics.adjusted_rand_score(labels, lab))
  print("Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels, lab))
  #print("Silhouette Coefficient: %0.3f"
  #  % metrics.silhouette_score(X_train_tfidf, lab, metric='sqeuclidean'))
  ###########################################

  ########### Agglomerative Clustering #######
  #model = AgglomerativeClustering(
  #          )
  #X_train_counts = X_train_counts.toarray()
  #model.fit(X_train_counts)
  #lab=model.labels_
  #ward = Ward().fit(X_train_lsa)
  #lab = np.reshape(ward.labels_)
  #print("Number of pixels: ", label.size)
  #print("Number of clusters: ", np.unique(label).size)
  #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, lab))
  #print("Completeness: %0.3f" % metrics.completeness_score(labels, lab))
  #print("V-measure: %0.3f" % metrics.v_measure_score(labels, lab))
  #print("Adjusted Rand Index: %0.3f"
  #  % metrics.adjusted_rand_score(labels, lab))
  #print("Adjusted Mutual Information: %0.3f"
  #  % metrics.adjusted_mutual_info_score(labels, lab))
  #print("Silhouette Coefficient: %0.3f"
  #  % metrics.silhouette_score(X_train_counts, lab, metric='sqeuclidean'))
  ###########################################


  ####### DBSCAN ############################
  #X_train_tfidf = X_train_tfidf.toarray()
  #X_train_tfidf = StandardScaler(with_mean=False).fit_transform(X_train_tfidf)
  #db = DBSCAN(eps=0.3, min_samples=10).fit(X_train_tfidf)
  #core_samples = db.core_sample_indices_    # should be an array
  #lab = db.labels_
  # Number of clusters in labels, ignoring noise if present.
  #n_clusters_ = len(set(lab)) - (1 if -1 in lab else 0)
  #print('Estimated number of clusters: %d' % n_clusters_)
  #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, lab))
  #print("Completeness: %0.3f" % metrics.completeness_score(labels, lab))
  #print("V-measure: %0.3f" % metrics.v_measure_score(labels, lab))
  #print("Adjusted Rand Index: %0.3f"
  #  % metrics.adjusted_rand_score(labels, lab))
  #print("Adjusted Mutual Information: %0.3f"
  #  % metrics.adjusted_mutual_info_score(labels, lab))
  #print("Silhouette Coefficient: %0.3f"
  #  % metrics.silhouette_score(X_train_tfidf, lab, metric='sqeuclidean'))
  #########################################

if __name__ == '__main__':
  main()