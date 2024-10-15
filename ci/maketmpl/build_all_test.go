package maketmpl

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildAll(t *testing.T) {
	tmp := t.TempDir()

	if err := BuildAll("testdata/BUILD.yaml", "testdata", tmp); err != nil {
		t.Fatal(err)
	}

	for _, tmpl := range []string{"reefy-ray", "fishy-ray"} {
		// Check that critical files are generated.
		for _, f := range []string{
			buildDotZip,
			rayAppDotJSON,
			readmeDocMD,
			readmeGitHubMD,
		} {
			stat, err := os.Stat(filepath.Join(tmp, tmpl, f))
			if err != nil {
				t.Errorf("os.Stat(%q): %v", f, err)
				continue
			}

			if stat.IsDir() {
				t.Errorf("%q is a directory", f)
			}
			if stat.Size() == 0 {
				t.Errorf("%q is empty", f)
			}
		}
	}
}
