# knn-undersampling

Weka package for KNN undersampling.

Based on code from here:

https://github.com/fracpete/knn-undersampling-weka-package

## Options

```
Filter options:

-k <number of references>
	Number of Nearest Neighbors (default 5).
-s <classname + options>
	Nearest Neighbors search method to use (default LinearNNSearch).
-t <Threshold decision>
	Threshold decision to remove, based in the count of neighbors belonging to another class (default 1).
-w <Index of majority class>
	Index of majority class, starting with 0 (default 0).
```

## Releases

* [2021.10.17](https://github.com/fracpete/knn-undersampling-weka-package/releases/download/v2021.10.17/knn-undersampling-2021.10.17.zip)


## Maven

Use the following dependency in your `pom.xml`:

```xml
    <dependency>
      <groupId>com.github.fracpete</groupId>
      <artifactId>knn-undersampling-weka-package</artifactId>
      <version>2021.10.17</version>
      <type>jar</type>
      <exclusions>
        <exclusion>
          <groupId>nz.ac.waikato.cms.weka</groupId>
          <artifactId>weka-dev</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
```


## How to use packages

For more information on how to install the package, see:

https://waikato.github.io/weka-wiki/packages/manager/


