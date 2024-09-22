package maketmpl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadmeFile_forGitHub(t *testing.T) {
	tmp := t.TempDir()

	content := strings.Join([]string{
		"# example",
		"",
		"![img1](img1.png)",
		"![img2](img2.png)",
		`<img src='img3.png' width=400px />`,
		"some extra text",
	}, "\n")

	path := filepath.Join(tmp, "readme.md")

	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal("write readme: ", err)
	}

	readme, err := readReadmeFile(path)
	if err != nil {
		t.Fatal("read readme: ", err)
	}

	output := filepath.Join(tmp, "readme.github.md")
	if err := readme.writeGitHubMD(output); err != nil {
		t.Fatal("write github md: ", err)
	}
}
