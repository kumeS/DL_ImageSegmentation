# GitHub from R
## Environment
- macOS High Sierra (10.13.6)
- R version 3.6.0 (2019-04-26)
- RStudio Version 1.2.1335

## Start git on MacOSX
```bash
#Install HomeBrew on Mac and set up the Brew.
brew update

# Install git via Homebrew
brew install git

# Check version
git --version
```

### then run RStudio :
```R
getwd()

setwd("[move to your directory]")

system("git config --global user.name [Your account name]")

system("git config --global user.email [Your email name]")

system("git config --global color.ui auto")

system("git config -l")

```

## Use Git from Rstudio
1. obtain your git repository. 
```R

system("git clone [Your Git https]")

```

2. check your config and status.
```R

setwd("[your git directory]")

system("ls -a")

system("git config -l")

system("git status")

```

3. edit files and folders

4. add: Add all changes to index
```R
# The below means " git add . & git add -u ".
system("git add -A")
```
5. commit: Register file in local repository
```R
system("git commit -m add new file")
```
6. push: Push the local repository and reflect it to the remote repository
```R
system("git push origin master")
```

### Save the password for https connection on RStudio
run Terminal below (if you need) :
```bash
git config --global credential.helper osxkeychain

# move to your local directory
git push origin master
# enter user account name and password 
# can use Git command from Rstudio
```

### To not create .DS_Store
```bash
defaults write com.apple.desktopservices DSDontWriteNetworkStores True

# Restart the Finder
killall Finder
```
