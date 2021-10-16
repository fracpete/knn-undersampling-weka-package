package weka.filters.supervised.instance;


import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.SimpleBatchFilter;
import weka.filters.SupervisedFilter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

/**
 * Filter for KNN undersampling.
 *
 * @author Marcel Beckmann (https://github.com/marcelobeckmann/knnund)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class KNNUndersampling
    extends SimpleBatchFilter
    implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

  private static final long serialVersionUID = -2103039882958523000L;

  /** Number of references */
  protected int m_k = 5;

  /** the threshold to use. */
  protected int m_Threshold = 1;

  /** the 0-based index of the majority class. */
  protected int m_MajorityClass = 1;

  /** the original instances. */
  public static Instances m_Original;

  /** for nearest-neighbor search. */
  protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

  /**
   * Returns the description of the classifier.
   *
   * @return description of the KNN class.
   */
  public String globalInfo() {
    return "In supervised learning, the imbalanced number of instances among the classes in a dataset can make the "
	+ "algorithms to classify one instance from the majority class as one from the majority class. With the aim "
	+ "to solve this problem, the KNN algorithm provides a basis to other balancing methods. These balancing "
	+ "methods are revisited in this work, and a new and simple approach of KNN undersampling is proposed. "
	+ "The experiments demonstrated that the KNN undersampling method outperformed other sampling methods. "
	+ "The proposed method also outperformed the results of other studies, and indicates that the simplicity "
	+ "of KNN can be used as a base for efficient algorithms in machine learning and knowledge discovery. \n\n"
	+ "For more information see:\n\n"
	+ getTechnicalInformation().toString();
  }

  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "Marcelo Beckmann, Nelson F. F. Ebecken, Beatriz S. L. Pires de Lima");
    result.setValue(TechnicalInformation.Field.TITLE, "A KNN Undersampling Approach for Data Balancing");
    result.setValue(TechnicalInformation.Field.YEAR, "2015");
    result.setValue(TechnicalInformation.Field.JOURNAL, "Journal of Intelligent Learning Systems and Applications");
    result.setValue(TechnicalInformation.Field.VOLUME, "7");
    result.setValue(TechnicalInformation.Field.PAGES, "104-116");
    result.setValue(TechnicalInformation.Field.URL, "http://dx.doi.org/10.4236/jilsa.2015.74010");

    return result;
  }

  /**
   * Sets the number of nearest neighbors to use.
   *
   * @param value new k value.
   */
  public void setK(int value) {
    m_k = value;
  }

  /**
   * Returns the number of nearest neighbors to use.
   *
   * @return the number of neighbors
   */
  public int getK() {
    return m_k;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String kTipText() {
    return "Number of Nearest Neighbors.";
  }

  /**
   * Sets the 0-based index of the majority class.
   *
   * @param value the index
   */
  public void setMajorityClass(int value) {
    this.m_MajorityClass = value;
  }

  /**
   * Returns the 0-based index of the majority class.
   *
   * @return the index
   */
  public int getMajorityClass() {
    return m_MajorityClass;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String majorityClassTipText() {
    return "Index of majority class, starting with 0.";
  }

  /**
   * Sets the nearest neighbor search to use.
   *
   * @param value the search
   */
  public void setNNSearch(NearestNeighbourSearch value) {
    m_NNSearch = value;
  }

  /**
   * Returns the nearest neighbor search to use.
   *
   * @return the search
   */
  public NearestNeighbourSearch getNNSearch() {
    return m_NNSearch;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String NNSearchTipText() {
    return "The nearest neighbor search method to use.";
  }

  /**
   * Sets the threshold.
   *
   * @param value the threshold
   */
  public void setThreshold(int value) {
    m_Threshold = value;
  }

  /**
   * Returns the threshold.
   *
   * @return the threshold
   */
  public int getThreshold() {
    return m_Threshold;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String thresholdTipText() {
    return "Threshold decision to remove, based in the count of neighbors belonging to another class (default 1).";
  }

  /**
   * Returns an enumeration of all the available options..
   *
   * @return an enumeration of all available options.
   */
  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();

    result.addElement(new Option(
	"\tNumber of Nearest Neighbors (default 5).",
        "k", 1, "-k <number of references>"));

    result.addElement(new Option(
        "\tNearest Neighbors search method to use (default LinearNNSearch).",
        "s", 1, "-s <classname + options>"));

    result.addElement(new Option(
	"\tThreshold decision to remove, based in the count of neighbors belonging to another class (default 1).",
	"t", 1, "-t <Threshold decision>"));

    result.addElement(new Option(
	"\tIndex of majority class, starting with 0 (default 0).",
        "w", 1, "-w <Index of majority class>"));

    return result.elements();
  }

  /**
   * Sets the OptionHandler's options using the given list. All options
   * will be set (or reset) during this call (i.e. incremental setting
   * of options is not possible).
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String option;
    String[] tmpOptions;
    String clsname;

    option = Utils.getOption('k', options);
    if (!option.isEmpty())
      setK(Integer.parseInt(option));
    else
      setK(5);

    option = Utils.getOption('s', options);
    if (!option.isEmpty()) {
      tmpOptions = Utils.splitOptions(option);
      clsname = tmpOptions[0];
      tmpOptions[0] = "";
      setNNSearch((NearestNeighbourSearch) Utils.forName(NearestNeighbourSearch.class, clsname, tmpOptions));
    }
    else {
      setNNSearch(new LinearNNSearch());
    }

    option = Utils.getOption('t', options);
    if (!option.isEmpty())
      setThreshold(Integer.parseInt(option));
    else
      setThreshold(0);

    option = Utils.getOption('w', options);
    if (!option.isEmpty())
      setMajorityClass(Integer.parseInt(option));
    else
      setMajorityClass(0);

    super.setOptions(options);
  }

  /**
   * Gets the current option settings for the OptionHandler.
   *
   * @return the list of current option settings as an array of strings
   */
  @Override
  public String[] getOptions() {
    List<String> result;

    result = new ArrayList<String>();

    result.add("-k");
    result.add("" + getK());

    result.add("-s");
    result.add("" + Utils.toCommandLine(getNNSearch()));

    result.add("-t");
    result.add("" + getThreshold());

    result.add("-w");
    result.add("" + getMajorityClass());

    result.addAll(Arrays.asList(super.getOptions()));

    return result.toArray(new String[0]);
  }

  /**
   * Returns the Capabilities of this filter. Derived filters have to override
   * this method to enable capabilities.
   *
   * @return the capabilities of this object
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.RELATIONAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Determines the output format based on the input format and returns this. In
   * case the output format cannot be returned immediately, i.e.,
   * immediateOutputFormat() returns false, then this method will be called from
   * batchFinished().
   *
   * @param inputFormat the input format to base the output format on
   * @return the output format
   * @throws Exception in case the determination goes wrong
   */
  @Override
  protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
    return inputFormat;
  }

  /**
   * Processes the given data (may change the provided dataset) and returns the
   * modified version. This method is called in batchFinished().
   *
   * @param data the data to process
   * @return the modified data
   * @throws Exception in case the processing goes wrong
   */
  @Override
  protected Instances process(Instances data) throws Exception {
    m_Original = new Instances(data);

    List <Integer> toBeRemoved = obtainInstancesToRemove(data);

    Instances cleanData = new Instances(data,0);
    for (int i = 0; i < data.numInstances(); i++) {
      if (!toBeRemoved.contains(i)) {
	Instance instance = data.instance(i);
	cleanData.add(instance);
      }
    }

    return cleanData;
  }

  /**
   * Determines instances to be removed.
   *
   * @param data the data to use
   * @return the instances to remove
   */
  protected List <Integer> obtainInstancesToRemove(Instances data) {
    // Obtain the samples from class w
    Instances majority = new Instances(data, 0);
    Enumeration en = data.enumerateInstances();
    while (en.hasMoreElements()) {
      Instance instance = (Instance) en.nextElement();
      if (instance.classValue() == m_MajorityClass) {
	majority.add(instance);
      }
    }

    // Instances for synthetic samples
    List <Integer> toRemove = new ArrayList();

    /*
     * Compute k nearest neighbors for i, and save the indices in the
     * nnarray
     */

    try {
      m_NNSearch.setInstances(data);
    }
    catch (Exception e) {
      e.printStackTrace();
    }

    en = data.enumerateInstances();
    int i = 0;
    while (en.hasMoreElements()) {
      Instance instance = (Instance) en.nextElement();
      if (instance.classValue() == m_MajorityClass) {
	List <Instance>knnList = generateKnnList(instance);
	if (decideToRemove( knnList)) {
	  toRemove.add(i);
	}
      }
      i++;
    }

    return toRemove;
  }

  /**
   * Function to take a decision about remove or not the instance
   */
  protected boolean decideToRemove(List<Instance> knnList) {
    int numberFrommajorityClasses = 0;
    for (int j = 0; j < knnList.size(); ++j) {
      Instance neighbor = knnList.get(j);
      if (neighbor.classValue() != m_MajorityClass)
	numberFrommajorityClasses++;
    }
    // TODO HOW TO DECIDE IF NOT ALL NEIGHBORS ARE FROM majority?
    return  (numberFrommajorityClasses>= m_Threshold);
  }

  protected List<Instance> generateKnnList(Instance instance) {
    List knnList = new ArrayList();
    try {
      Instances nns = m_NNSearch.kNearestNeighbours(instance, m_k);
      for (int i = 0; i < nns.numInstances(); i++) {
	knnList.add(nns.instance(i));
      }
      return knnList;
    }
    catch (Exception e) {
      e.printStackTrace();
      return null;
    }
  }

  /**
   * Returns the revision string.
   *
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv
   *            should contain arguments to the filter: use -h for help
   */
  public static void main(String[] argv) {
    runFilter(new KNNUndersampling(), argv);
  }
}
