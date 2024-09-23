package maketmpl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadmeFile_writeGitHubMD(t *testing.T) {
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

	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal("read output: ", err)
	}

	want := strings.Join([]string{
		"# example",
		"",
		`<img src="img1.png" alt="img1" />`,
		`<img src="img2.png" alt="img2" />`,
		`<img src="img3.png" width="400px" />`,
		"some extra text",
	}, "\n")

	if string(got) != want {
		t.Errorf("got:\n---\n%s\n---\nwant:\n---\n%s\n---\n", got, want)
	}
}

func TestReadmeFile_writeReleaseMD(t *testing.T) {

}
