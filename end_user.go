package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
)

// Number of customers shown in the chart, highest retention uplift first.
const topN = 20

// One customer from outputs/customer_recommendations.csv (written by the notebook).
type ChurnData struct {
	CustomerID      string
	RetentionUplift float64
}

func main() {
	path := filepath.Join("outputs", "customer_recommendations.csv")
	http.HandleFunc("/", dashboardHandler(path))
	log.Println("View your Churn Dashboard at http://localhost:8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
}

// dashboardHandler re-reads the CSV on every request, so the chart reflects the
// latest notebook run.
func dashboardHandler(path string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := readCSV(path)
		if err != nil {
			log.Printf("dashboard: %v", err)
			http.Error(w, fmt.Sprintf("%v\nRun Churn_Uplift_COMPLETE.ipynb first, "+
				"then start this server from the repository root.", err), http.StatusInternalServerError)
			return
		}
		data = topByUplift(data, topN)

		bar := charts.NewBar()
		bar.SetGlobalOptions(
			charts.WithTitleOpts(opts.Title{Title: "Customer Uplift Analysis",
				Subtitle: fmt.Sprintf("Top %d customers by retention uplift (higher = offer keeps the customer)", len(data))}),
			charts.WithTooltipOpts(opts.Tooltip{Show: opts.Bool(true)}),
		)

		categories := make([]string, 0, len(data))
		values := make([]opts.BarData, 0, len(data))
		for _, item := range data {
			categories = append(categories, item.CustomerID)
			values = append(values, opts.BarData{Value: item.RetentionUplift})
		}
		bar.SetXAxis(categories).AddSeries("Retention uplift", values)

		if err := bar.Render(w); err != nil {
			log.Printf("dashboard: rendering chart: %v", err)
		}
	}
}

// readCSV loads customer IDs and retention uplift, finding the columns by
// header name so column order in the notebook output does not matter.
func readCSV(filename string) ([]ChurnData, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("%s: reading header: %w", filename, err)
	}
	idCol, upliftCol := -1, -1
	for i, name := range header {
		switch name {
		case "customer_index":
			idCol = i
		case "retention_uplift":
			upliftCol = i
		}
	}
	if idCol < 0 || upliftCol < 0 {
		return nil, fmt.Errorf("%s: needs customer_index and retention_uplift columns", filename)
	}

	var results []ChurnData
	for {
		line, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filename, err)
		}
		uplift, err := strconv.ParseFloat(line[upliftCol], 64)
		if err != nil {
			return nil, fmt.Errorf("%s: customer %s: %w", filename, line[idCol], err)
		}
		results = append(results, ChurnData{CustomerID: line[idCol], RetentionUplift: uplift})
	}
	return results, nil
}

// topByUplift returns the n customers with the highest retention uplift,
// leaving the input slice untouched.
func topByUplift(data []ChurnData, n int) []ChurnData {
	sorted := append([]ChurnData(nil), data...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].RetentionUplift > sorted[j].RetentionUplift
	})
	if len(sorted) > n {
		sorted = sorted[:n]
	}
	return sorted
}
