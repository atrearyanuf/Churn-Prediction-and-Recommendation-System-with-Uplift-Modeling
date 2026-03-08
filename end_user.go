package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
)

// Data structure to hold our CSV rows
type ChurnData struct {
	CustomerID  string
	UpliftScore float64
}

func main() {
	http.HandleFunc("/", dashboardHandler)
	log.Println("View your Churn Dashboard at http://localhost:8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	// 1. Read the Python-generated CSV
	data := readCSV("outputs\\customer_recommendations.csv")

	// 2. Create a Bar Chart for Uplift Scores
	bar := charts.NewBar()
	bar.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{Title: "Customer Uplift Analysis",
			Subtitle: "Which customers should we target?"}),
		charts.WithTooltipOpts(opts.Tooltip{Show: opts.Bool(true)}),
	)

	// Prepare data for the chart
	categories := []string{}
	values := []opts.BarData{}
	for _, item := range data {
		categories = append(categories, item.CustomerID)
		values = append(values, opts.BarData{Value: item.UpliftScore})
	}

	bar.SetXAxis(categories).AddSeries("Uplift Score", values)

	// 3. Render the chart as HTML to the browser
	bar.Render(w)
}

func readCSV(filename string) []ChurnData {
	f, _ := os.Open(filename)
	defer f.Close()

	reader := csv.NewReader(f)
	var results []ChurnData

	// Skip header
	reader.Read()

	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		}

		// Convert string score to float
		score := 0.0
		if len(line) > 1 {
			fmt.Sscanf(line[1], "%f", &score)
		}

		results = append(results, ChurnData{CustomerID: line[0], UpliftScore: score})
	}
	return results
}
