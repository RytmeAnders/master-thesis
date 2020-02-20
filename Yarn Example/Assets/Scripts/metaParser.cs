using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace Yarn.Unity {
    public class metaParser : MonoBehaviour
    {
        // The dialogue runner that we want to attach the 'visited' function to
        [SerializeField] Yarn.Unity.DialogueRunner dialogueRunner;

        [YarnCommand("printMeta")]
        public void PrintMeta()
        {
            string filepath = "data.csv";
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@filepath, true))
            {
                Debug.Log("Test: ");
                file.WriteLine(Time.time + "," + dialogueRunner.currentNodeName);
            }
        }

        public void PrintOption(string playerType)
        {
            string filepath = "data.csv";
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@filepath, true))
            {
                Debug.Log("Printing" + playerType);
                file.WriteLine(Time.time + "," + playerType);
            }
        }

        public void LoadQuestionnaire()
        {
            Application.OpenURL("https://matthewbarr.co.uk/bartle/");
        }

    }
}
