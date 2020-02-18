﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace Yarn.Unity {
    public class metaParser : MonoBehaviour
    {
        // The dialogue runner that we want to attach the 'visited' function to
        [SerializeField] Yarn.Unity.DialogueRunner dialogueRunner;

        [YarnCommand("printMeta")]
        public void Printing()
        {
            string filepath = "data.csv";
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@filepath, true))
            {
                Debug.Log("Test: ");
                file.WriteLine(Time.time + "," + dialogueRunner.currentNodeName);
            }
        }
    }
}
