How to make a release
=====================

Preparation
-----------

* Change the artifact ID in `pom.xml` to today's date, e.g.:

  ```
  2021.10.17-SNAPSHOT
  ```

* Update the version, date and URL in `Description.props` to reflect new
  version, e.g.:

  ```
  Version=2021.10.17
  Date=2021-10-17
  PackageURL=https://github.com/fracpete/knn-undersampling-weka-package/releases/download/v2021.10.17/knn-undersampling-2021.10.17.zip
  ```

* Commit/push all changes


Weka package
------------

* Run the following command to generate the package archive for version `2021.10.17`:

  ```
  ant -f build_package.xml -Dpackage=knn-undersampling-2021.10.17 clean make_package
  ```

* Create a release tag on github (v2021.10.17)
* add release notes
* upload package archive from `dist`


Maven
-----

* Run the following command to deploy the artifact:

  ```
  mvn release:clean release:prepare release:perform
  ```

* log into https://oss.sonatype.org and close/release artifacts

* After successful deployment, push the changes out:

  ```
  git push
  ````

