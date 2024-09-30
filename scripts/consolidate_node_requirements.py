import requests
import json
import sys
import argparse
import re
from packaging.requirements import Requirement, InvalidRequirement
from packaging.specifiers import SpecifierSet
from urllib.parse import urljoin, urlparse
import time

def get_default_branch(repo_url, headers=None):
    # Extract owner and repo name from the URL
    path_parts = urlparse(repo_url).path.strip('/').split('/')
    if len(path_parts) < 2:
        print(f"Invalid repository URL: {repo_url}")
        return None
    owner, repo_name = path_parts[:2]
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"

    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            repo_info = response.json()
            return repo_info.get('default_branch', 'main')
        else:
            print(f"Failed to fetch default branch for {repo_url}: HTTP {response.status_code}")
            return 'main'  # Fallback to 'main' if we can't fetch the default branch
    except Exception as e:
        print(f"Error fetching default branch for {repo_url}: {e}")
        return 'main'  # Fallback to 'main' in case of error

def process_requirements_from_repo(repo_url, commit_hash, dependency_dict, url_requirements, visited_repos, headers=None):
    if not repo_url:
        return

    # Avoid processing the same repository and commit multiple times
    repo_commit_key = (repo_url, commit_hash)
    if repo_commit_key in visited_repos:
        return
    visited_repos.add(repo_commit_key

    )

    if not commit_hash:
        # Get the default branch name
        commit_hash = get_default_branch(repo_url, headers=headers)
        if not commit_hash:
            print(f"Skipping repository due to missing commit and default branch: {repo_url}")
            return

    # Construct the raw URL to the requirements.txt file at the specified commit or branch
    raw_base_url = repo_url.replace("github.com", "raw.githubusercontent.com")
    raw_requirements_url = urljoin(raw_base_url + "/", f"{commit_hash}/requirements.txt")

    try:
        response = requests.get(raw_requirements_url, headers=headers)
        if response.status_code == 200:
            requirements = response.text.splitlines()
            for req_line in requirements:
                req_line = req_line.strip()
                if not req_line or req_line.startswith('#'):
                    continue
                try:
                    req = Requirement(req_line)
                    name = req.name.lower()
                    specifier = req.specifier
                    # Resolve version conflicts
                    if name in dependency_dict:
                        existing_specifier = dependency_dict[name]
                        # Use the intersection of specifiers
                        combined_specifier = existing_specifier & specifier
                        dependency_dict[name] = combined_specifier
                    else:
                        dependency_dict[name] = specifier
                except InvalidRequirement:
                    # Cannot parse the requirement, check if it's a git+ URL
                    if req_line.startswith('git+'):
                        # Extract the repository URL from the git+ requirement
                        git_repo_url, git_commit = parse_git_requirement(req_line)
                        if git_repo_url and git_repo_url not in visited_repos:
                            process_requirements_from_repo(git_repo_url, git_commit, dependency_dict, url_requirements, visited_repos, headers=headers)
                    else:
                        # Include unparseable requirement as-is
                        url_requirements.add(req_line)
                except Exception as e:
                    print(f"Unexpected error parsing requirement '{req_line}' from {repo_url}: {e}")
        else:
            print(f"No requirements.txt found in {repo_url} at {commit_hash} (HTTP {response.status_code})")
    except Exception as e:
        print(f"Error fetching requirements.txt from {repo_url}: {e}")

    # Respect GitHub API rate limits by sleeping briefly (optional)
    time.sleep(0.1)  # Sleep for 0.1 seconds between requests to avoid hitting rate limits

def parse_git_requirement(req_line):
    """
    Parses a git+ requirement line and extracts the repository URL and commit hash if specified.
    """
    pattern = r'git\+(https://[^@]+)(?:@([^#]+))?(?:#egg=([\w\d\.]+))?'
    match = re.match(pattern, req_line)
    if match:
        repo_url = match.group(1)
        commit_hash = match.group(2)
        # Remove .git suffix from URL if present
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        return repo_url, commit_hash
    else:
        return None, None

def consolidate_requirements(json_file):
    # Optionally, set up headers with GitHub token to avoid rate limits
    headers = {}
    # headers = {'Authorization': 'token YOUR_GITHUB_TOKEN'}

    # Load repositories from the JSON file
    try:
        with open(json_file, 'r') as f:
            repos = json.load(f)
        if not isinstance(repos, list):
            raise ValueError("JSON file must contain a list of repositories.")
    except Exception as e:
        print(f"Error reading JSON file '{json_file}': {e}")
        sys.exit(1)

    dependency_dict = {}
    url_requirements = set()
    visited_repos = set()

    for repo in repos:
        repo_url = repo.get("repo")
        commit_hash = repo.get("commit", "").strip()
        if not repo_url:
            print(f"Invalid repository entry (missing 'repo' key): {repo}")
            continue

        process_requirements_from_repo(repo_url, commit_hash, dependency_dict, url_requirements, visited_repos, headers=headers)

    # Build the consolidated requirements list
    consolidated_requirements = []
    for name, specifier in dependency_dict.items():
        if specifier:
            consolidated_requirements.append(f"{name}{specifier}")
        else:
            consolidated_requirements.append(name)

    # Write the consolidated requirements to a file
    with open('requirements.txt', 'w') as f:
        for req in sorted(consolidated_requirements):
            f.write(f"{req}\n")

    print("Consolidated requirements.txt has been generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate requirements.txt files from multiple repositories.')
    parser.add_argument('json_file', help='Path to the JSON file containing repository information.')
    args = parser.parse_args()
    consolidate_requirements(args.json_file)
