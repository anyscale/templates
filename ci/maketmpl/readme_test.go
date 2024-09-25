package maketmpl

import (
	"errors"
	"os"
	"os/exec"
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
	tmp := t.TempDir()

	content := strings.Join([]string{
		"# example",
		"",
		`![img1](img1.png)<img src='img2.png' width=400px />extra text`,
	}, "\n")

	for _, file := range []struct {
		name, content string
	}{
		{"img1.png", "img1"},
		{"img2.png", "img2"},
		{"readme.md", content},
	} {
		path := filepath.Join(tmp, file.name)
		if err := os.WriteFile(path, []byte(file.content), 0600); err != nil {
			t.Fatalf("write %q: %s", file.name, err)
		}
	}

	input := filepath.Join(tmp, "readme.md")
	readme, err := readReadmeFile(input)
	if err != nil {
		t.Fatal("read readme: ", err)
	}

	output := filepath.Join(tmp, "readme.release.md")
	if err := readme.writeReleaseMD(output, tmp); err != nil {
		t.Fatal("write release md: ", err)
	}

	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal("read output: ", err)
	}

	want := strings.Join([]string{
		"# example",
		"",
		strings.Join([]string{
			`<img src="data:image/png;base64,aW1nMQ==" alt="img1" />`,
			`<img src="data:image/png;base64,aW1nMg==" style="width: 400px" />`,
			`extra text`,
		}, ""),
	}, "\n")

	if string(got) != want {
		t.Errorf("got:\n---\n%s\n---\nwant:\n---\n%s\n---\n", got, want)
	}
}

func TestReadmeFromNotebook(t *testing.T) {
	if _, err := exec.LookPath("jupyter"); err != nil {
		if errors.Is(err, exec.ErrNotFound) && os.Getenv("CI") == "" {
			t.Skip("jupyter not found; skip the test as it is not on CI.")
		}
	}

	tmp := t.TempDir()

	f, err := readmeFromNotebook("testdata/readme.ipynb")
	if err != nil {
		t.Fatal("read readme from notebook: ", err)
	}

	output := filepath.Join(tmp, "readme.github.md")
	if err := f.writeGitHubMD(output); err != nil {
		t.Fatal("write github md: ", err)
	}

	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal("read output: ", err)
	}

	want := strings.Join([]string{
		"# Test example",
		"",
		`<img src="a.png" width="400px" />`,
		"",
		"and some text",
		"",
		"",
		"```python",
		`print("this is just an example")`,
		"```",
		"",
	}, "\n")

	if string(got) != want {
		t.Errorf("got:\n---\n%s\n---\nwant:\n---\n%s\n---\n", got, want)
	}
}
